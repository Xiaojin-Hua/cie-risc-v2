# 任务：修 `n_rs_seq > 0` 时模型输出错误（fork 层正确性 bug）

## 症状

同一个 prompt、同一份权重：

| `n_rs_seq` | 输出 |
|---|---|
| **0** | `RISC-V 向量扩展是指通过增加指令集…`（正确） |
| **1 / 2 / 3 / 4** | `R <think>…`（第一个 token 就错），随后退化成重复空格 |

不是草稿模型的问题 —— 用 `ngram-simple`（没有草稿模型）也能复现。

## 影响（为什么这是最高优先级）

投机解码必须开 `n_rs_seq`（回滚未被接受的草稿 token 需要循环状态快照），
而 llama-server 的投机路径把 `n_rs_seq` 设成 `draft.n_max`。因此：

- **所有投机路径的输出都是错的**
- 所有已测的投机接受率（88.5% / 79.5% / 44.4% 等）都是在错误前向上得到的，
  重复空格的输出让草稿变得极易猜中 → 接受率虚高
- 关掉 `n_rs_seq`（lossless）后：2B + 重训草稿的接受率掉到 **38%**，速度 **1.90 t/s**（基线 5.25）

## 已经排除的可能

**ubatch 切分本身没问题**：`llama-perplexity -ub 512 / 128 / 16` 得到
PPL = 13.1725 / 13.1713 / 13.1465（差 0.2%，属于浮点累加顺序差异）。

所以 bug 不在"批被切小"，而在 `n_rs_seq` 打开的**状态快照机制**。

## 精确定位（已确认的代码路径）

**1. 参数定义** `src/llama-cparams.h:16`

```cpp
uint32_t n_rs_seq;   // number of recurrent-state snapshots per seq for rollback
```

**2. 进入 GDN 图构建** `src/models/delta-net-base.cpp:590`

```cpp
const int64_t K = cparams.n_rs_seq + 1;
ggml_tensor * gdn_out = ggml_gated_delta_net(ctx0, q, k, v, g, b, s, nullptr, nullptr, K);
```

**3. 缓冲区布局**

```
gdn_out = [ 每个 token 的输出 (attn_score_elems) ][ K × state_size_per_snap ]
```

- `ggml_view_4d(ctx0, gdn_out, S_v, H_v, n_seq_tokens, n_seqs, …)` → 取输出（偏移 0）
- `ggml_view_3d(ctx0, gdn_out, D, n_seqs, n_written, …)` → 取快照，写回 recurrent cache
- `n_written = min(n_seq_tokens, K)`，注释说"trailing slots are left unwritten"

**4. 怀疑点**

`ggml/src/ggml-cpu/ops.cpp` 里 `ggml_gated_delta_net` 的 **`K > 1` 分支**。
最可能是：写快照时踩到了正在迭代的运行状态，或快照偏移算错，导致后续 token 全部算错。

**注意**：同一个文件里已经修过一个同类 bug（conv-state 的 4D 索引），
所以这个 fork 的 batched/GDN 路径存在成体系的移植不完整问题，值得系统看一遍。

## 复现方式（最快）

llama-bench 已被改造支持投机，带 `--spec-rs-seq`：

```bash
B=llama.cpp/build-k1/bin
export LD_LIBRARY_PATH=$B

# 正确输出
$B/llama-bench -m /home/bianbu/kwork/nongemm/ssm_all_k128.gguf -t 4 -p 128 -n 32 -fa 1 \
    --spec-type draft-eagle3 -md /home/bianbu/kwork/specdecode/e3n_q4.gguf --spec-rs-seq 0

# 错误输出
... --spec-rs-seq 1
```

也可以用 `llama-perplexity`（如果它暴露了 `n_rs_seq` 参数）：
`--spec-rs-seq 0` 与 `1` 的 PPL 应当**完全相同**，现在不是。

## 调试方法

1. 固定 prompt 和 batch，`rs_seq=0` vs `1`，比较 `llama_get_logits` —— 确认从第几个 token 开始不同
2. 若 logits 已不同：逐层比较 hidden state，找**第一个**出现差异的层
3. 该层必然是 GDN（线性注意力）层；进入 `ggml_gated_delta_net` 的 `K > 1` 分支
4. 对比 `K=1` 与 `K=2` 时：
   - 每个 token 的输出是否逐位一致（`argmax` 相同还不够，要看数值）
   - 最终写回的状态是否一致
   - 第 i 份快照是否等于"处理完第 i 个 token 时的状态"
5. 常见错误形态：快照写到了 `gdn_out` 的输出区（覆盖了 attn_output）、
   或状态写回用了快照区的偏移、或 `n_written` 计算让最后一个 token 的状态没写回

## 已知的坑（都踩过）

1. **不要构建 `llama-server`**：它的 CMake 目标会去 HuggingFace 下载 Web UI，
   板子无外网会卡死十几分钟。只构建需要的 target（如 `llama-bench`、`llama-perplexity`）。
   若改了 `common/`，先 `--target llama-common`。
2. **编译器必须 gcc-14 / g++-14**（默认 gcc 13.2 缺 `vfloat16m2_t`）。
3. **板子有残余负载**：先 `cat /proc/loadavg`（理想 < 1.5），必要时 `taskset` 绑核；
   跑前 `pkill -9 -x llama-server`。
4. **不要 commit，不要 `git reset --hard`**，改动留在工作区让用户 review。

## 验收标准

1. `--spec-rs-seq 1` 的输出与 `--spec-rs-seq 0` **逐 token 相同**
2. `llama-perplexity` 在 `n_rs_seq=0 / 1` 下 PPL 相同（差 < 0.01）
3. 基线 `tg128` 不漂移（仍 5.2–5.35 t/s）
4. 修好后，重新测两个草稿的接受率（都在 lossless 前提下），给出**可信**对比：
   - `/home/bianbu/kwork/specdecode/e3g_allq4.gguf`（原版，HF 特征训练）
   - `/home/bianbu/kwork/specdecode/e3n_q4.gguf`（重训，100 万板端 token）

## 资产

| 文件 | 说明 |
|---|---|
| `/home/bianbu/kwork/nongemm/ssm_all_k128.gguf` | 2B 目标模型（Q4_0_K128） |
| `/home/bianbu/kwork/specdecode/e3n_q4.gguf` | 重训草稿（q4_0，104 MB） |
| `/home/bianbu/kwork/specdecode/e3g_allq4.gguf` | 原始草稿（q4_0，87 MB） |
| `/home/bianbu/cie-risc-v2/tools/spec_bench/` | 测量工具（evalscope 驱动器、判据文档） |

## 背景数字（**都不可信**，仅供定位参考）

| 配置 | 来源 | 数字 |
|---|---|---|
| 基线 tg128 | llama-bench | 5.25 t/s |
| 2B + e3n，rs_seq 默认（坏） | llama-server | 88.46% 接受率 / mean len 3.63 / 192.75 ms/token |
| 2B + e3n，rs_seq=0（lossless） | llama-bench | 38% / 1.90 / 1.90 t/s |
