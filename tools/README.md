# tools — 权重转换与 CED 训练

两块内容，各自独立：

| 目录 | 作用 |
|---|---|
| `tools/weight_transfer/` | 权重转化：任意 HF/高精度 GGUF → 板端 `Q4_0_K128` GGUF（一键） |
| `tools/ced/` | CED（Cross-layer Encoder-Decoder）K/V 投影的训练、导出与校验 |

---

# 1. 权重转化

## 1.1 一键用法

```bash
export LLAMA_QUANTIZE=/path/to/llama.cpp/build/bin/llama-quantize   # 或设 LLAMA_ROOT
./tools/weight_transfer/ckpt_transfer.sh <source_path> <output_path>
```

* `<source_path>`：HF 模型目录（`config.json` + `*.safetensors`），或**高精度** GGUF（f16 / bf16 / q8_0）
* `<output_path>`：输出的 `.gguf`（`Q4_0_K128`）

产出与这两个参考模型**布局一致**：

| 模型 | 文件 | 大小 |
|---|---|---:|
| Qwen3.5-2B | `ssm_all_k128.gguf` | 1,007,028,256 B |
| Qwen3.5-4B | `Qwen3.5-4B-k128-v2.gguf` | 2,212,707,232 B |

## 1.2 四步流水线（**第 3、4 步都不能省**）

```
1. HF 目录 → f16 GGUF              convert_hf_to_gguf.py
2. f16     → q4_0 base (K64)       llama-quantize --output-tensor-type q4_k
3. base + 高精度源 → K64(共享scale) requantize_k128.py --block 128
4. K64(成对) → 真正的 K128(type 44) pair_to_k128.py   ← 无损重排 68B→66B
```

**每一步为什么存在：**

1. **转 f16**：统一入口，后面两步都吃 GGUF。
2. **q4_0 base**：提供元数据和所有非 Q4_0 张量。`--output-tensor-type q4_k` 不能省——
   `output.weight` 是 `248320 × hidden` 的大张量（2B 508M / 4B 636M 参数），
   llama-quantize 对 q4_0 默认把它放成 **Q6_K**，而参考模型用的是 **Q4_K**；
   decode 每 token 都要读它，Q6_K 会白丢约 17% 的 tg 吞吐。
3. **requantize --block 128**：把 128 宽的重编码成"两个共享同一 fp16 scale 的 K64 记录"。
   **注意：这一步产出的文件类型仍是 K64(43)，只是数值等价 K128。**
   跳过第 4 步就会跑 K64 内核 —— block scale 开销翻倍，实测 pp 掉 18%。
4. **pair_to_k128**：把成对的 K64 记录无损重排成真正的 `Q4_0_K128`（type 44，66 字节/128 值）。

## 1.3 K128 块格式（对照 `ggml-quants.c: quantize_row_q4_0_k128_ref`）

```
每个 128 值一个块：ggml_half d + uint8_t qs[64]   （66 字节）
  amax = max|x|;  max = 取到 amax 的那个带符号值;  d = max / -8
  h = 0..3, j = 0..15:
      xi0 = min(15, int(x[h*32 + j]      * id + 8.5))
      xi1 = min(15, int(x[h*32 + j + 16] * id + 8.5))
      qs[h*16 + j] = xi0 | (xi1 << 4)
```

自检（会打印每个类型的张量数，布局不符返回非 0）：

```bash
python3 tools/weight_transfer/check_k128.py out.gguf
#   Q4_0_K128 (44) : 248      <- 每个可量化张量
#   Q4_K      (12) : 1        <- output.weight
#   F32        (0) : 177      <- norm/bias
#   OK
```

## 1.4 实测（Qwen3.5-4B，板端 t=4，`-fa 1`）

| 版本 | pp128 | pp512 | tg128 |
|---|---:|---:|---:|
| K64 + Q6_K（第 4 步与 output 类型都错） | 14.85 | 15.02 | 2.12 |
| K128 + Q6_K | 17.55 | 17.26 | 2.17 |
| **K128 + Q4_K（正确）** | **17.58** | **17.41** | **2.54** |

2B 参考：pp128 47.5 / tg128 5.35。

---

# 2. CED 训练

CED 不改模型权重。它训一个**旁路投影**（blob），运行时用
`LLAMA_QWEN35_CED=<blob>` 加载，重建前缀的 K/V。

## 2.1 原理

把模型按层切成两半：

```
encoder = layers [0, split)        每个 token 都过（精确）
decoder = layers [split, n_layer)  只过最后 tail 个 token
```

* **全注意力层**：decoder 里的 Q 只算 tail，但 K/V 需要覆盖整段前缀。
  K/V = `concat(前缀投影出的 K/V, tail 的精确 K/V)`。
* **GDN 层**（线性注意力）：只在 tail 上跑，循环/卷积状态从**零**开始。
* **前缀的 K/V 从哪来**：对 `H_split`（第 split 层的隐状态）做一个**低秩线性投影**。
  这就是训练的对象：每个被校准层一对 `K[kv_dim, hidden]` / `V[kv_dim, hidden]`。

**为什么能加速**：前缀 token 只走 encoder 一半的层，prefill 的 FLOPs 大幅下降；
tail（默认 128）走完整层。

脚本内置正确性自检：当 `tail == ctx` 时，CED 退化等价于整模型，
它会把 `max|dlogit| == 0` 打印出来（我们实测 0.00% 的 PPL 变化）。

## 2.2 训练数据

**格式**：token 分片 + 索引 JSON。

```
data/corpus/
├── index.json          {"shards": [{"path": "shard_000.npy"}, ...]}
├── shard_000.npy       int32 一维数组（token id）
├── shard_001.npy
└── shard_002.npy       ← 最后一个分片当作 held-out eval
```

训练器只做 `np.concatenate(np.load(前 n-1 个))` 和 `np.load(最后一个)`，
所以分片就是一个纯 token 数组，**不需要任何额外元数据**。

**生成方式**（任意纯文本语料都可以）：

```bash
python3 tools/ced/gen_ced_corpus.py \
    --model <HF 模型目录>           \
    --text <语料.txt> [更多.txt]     \
    --out data/corpus --shard-tokens 2000000
```

**我们用的语料**：

| 来源 | 链接 | 说明 |
|---|---|---|
| wikitext-2-raw | https://huggingface.co/datasets/Salesforce/wikitext （`wikitext-2-raw-v1`） | 板上位置 `/home/bianbu/k64-precision/corpus/wikitext-2-raw/wiki.test.raw` |
| A100 上的现成语料 | `/home/gfx/pop-bridge/data/wiki_long.txt`（60k token）、`wiki24k.txt`（24k） | 平台只在做验证时够用 |

> 语料规模建议：**至少 2M token**、多个分片。我们 2B 验证时只用了 65k token（4 分片），
> 只能证明流程通，不能证明收敛质量。

## 2.3 训练

```bash
# 2B
./tools/ced/ced_train.sh --size 2b --out data/ced_2b --tag v1 --tail 128 --train-steps 400 --windows 8

# 4B
./tools/ced/ced_train.sh --size 4b --out data/ced_4b --tag v1 --tail 128 --train-steps 400 --windows 8
```

`--size` 选的那套参数（其余全部透传给训练器）：

| | 2B | 4B |
|---|---:|---:|
| 模型默认路径 | `/home/gfx/pop-bridge/models/Qwen3.5-2B-Base` | `.../Qwen3.5-4B` |
| hidden | 2048 | 2560 |
| kv_dim | 512（2 头 × 256） | 1024（4 头 × 256） |
| layers | 24 | 32 |
| **split_layer** | **14** | **17** |
| tail / ctx | 512 / 512（可覆盖） | 512 / 512 |

**split_layer 必须落在 GDN 层上**：全注意力层的位置是 `4k+3`，
2B 是 3/7/11/15/19/23，4B 是 3/7/11/15/**19**/23/27/31。
2B 取 14 恰好让 decoder 以 GDN 层开头；4B 取 **17**（不是按比例算的 19），
否则 decoder 会从全注意力层开始，CED 两条路径的切换点就和 2B 不同构了。

常用可调项（透传）：

| 参数 | 含义 |
|---|---|
| `--train-steps N` | 梯度步数；`0` = 只测量不训练（用于复现基准） |
| `--windows N` | 每轮采样多少个 ctx 窗口 |
| `--capacity N` | 投影用 MLP 隐层（0 = 单层线性，**blob 只支持线性**） |
| `--lr / --batch` | 学习率 / 批大小 |
| `--train-tails` | 训练用的多个 tail 长度，避免过拟合单一注意力模式 |

## 2.4 导出与校验

```bash
python3 tools/ced/export_ced_blob.py \
    --src data/ced_4b/v1.npz  --out ced_proj_4b.bin \
    --hidden 2560 --kv 1024 --split 17 --tail 128
```

blob 格式（板端 `src/models/llama-ced.h` + `tools/ced/check_export.py` 双向确认）：

```
offset 0 : char   magic[8] = "CEDPROJ1"
offset 8 : uint32 n_layers_calibrated
offset 12: uint32 hidden_size
offset 16: uint32 kv_dim
offset 20: uint32 split_layer
offset 24: uint32 tail
offset 28: uint32 layer_ids[n]
之后每层 : fp16 K[kv_dim, hidden]  然后  fp16 V[kv_dim, hidden]
```

尺寸可自检：2B（3 层）= `28 + 12 + 3*2*512*2048*2` = 12,582,952 B；
4B（4 层）= `28 + 16 + 4*2*1024*2560*2` = 41,943,084 B。

`check_export.py` 会重跑 GPU 侧的投影并与 blob 逐元素比对，
防止转置/缩放错误（板端对这两者都不会报错，只会静默出错）。

## 2.5 实测结果

| 模型 | blob | pp128 无 CED | pp128 有 CED | 提升 |
|---|---|---:|---:|---:|
| 2B | `ced_proj_2b.bin`（tail 128，3 层） | 47.5 | 65.5（tail 16） | **+38%** |
| 4B | `ced_proj_4b.bin`（tail 128，split 17，4 层） | 17.58 | **24.59** | **+39.9%** |

精度（2B，160 题中文公务员考试，`enable_thinking=false`）：
无 CED 74/160、CED 74/160 —— **无差异**。

## 2.6 坑（都踩过）

1. **CED 小 tail 的 PPL 无意义**：`llama-perplexity` 从 `n_ctx/2` 开始打分，
   而 CED 只对末尾 `tail` 个 token 出 logits。`tail <= 16` 时打分区间几乎全在窗口外，
   数字是垃圾（见过 53697）。必须用 `--ppl-first` 或走准确率评测。
2. **CED 与 EAGLE3 不兼容**：CED 不为 `split` 之后的层产出 `t_layer_inp`，
   而 EAGLE3 的 `target_layers` 含第 23 层 ⇒ `set_outputs` 断言失败。跑 EAGLE3 要 `env -u LLAMA_QWEN35_CED`。
3. **blob 与模型绑定**：`hidden / kv_dim / split` 都写在头里，2B 的 blob 不能给 4B 用。
4. **`tail` 是绝对 token 数**：n=128 时 `tail=16` 会剪掉 87.5%，`tail=64` 只剪一半，
   pp 速度从 65 掉到 55 —— 短 prompt 上这个参数主导一切。
5. **投影只能是线性**（`--capacity 0`）：板端按 `K @ h` 单次矩阵乘实现，
   带 MLP 隐层的投影导出不了。
