# 让投机解码转正：verify 阶段只算草稿词表（d2t）的 logits

## 问题（实测）

| | 每 token | 每轮（m=4 verify） |
|---|---:|---:|
| 基线 | 174 ms | 174 ms |
| ngram 投机（len 2.10） | 202 ms | **422 ms** |

`(422 − 174) / 3 = 83 ms` —— 这是**每个额外 verify 位置的边际成本**。

归因：Qwen3.5 词表 **248,320**，输出层 `248k × hidden` 每个位置约 1 GFLOP（2B）。
verify 需要每个位置的 argmax 才能判接受，所以

```
verify(m) ≈ 各层前向(≈106 ms，与 m 近无关) + m × 输出头(≈79 ms)
```

## 洞察

草稿只能提出 **d2t（32k 草稿词表）**内的 token，C++ 侧（`common/speculative.cpp`）
本来就是把 32k logits 散射进 248k 再取最大。**所以 verify 只需要 32k 行的 logits。**

输出头成本可降 248k/32k = **7.75×**：

| | 现在 | 改后 |
|---|---:|---:|
| verify(4) | 106 + 4×79 = 422 ms | 106 + 4×10 = **146 ms** |
| 每 token（len 2.10） | 201 ms | **70 ms** |
| 吞吐 | 4.79 t/s（−2.4%） | **≈14 t/s（+180%）** |

对 EAGLE3（k=3）同样有效；4B 上收益更大（每位置 1.3 GFLOP）。

## 实现位置

| # | 文件 | 改动 |
|---|---|---|
| 1 | `src/models/qwen35.cpp:727` | `build_lora_mm(model.output, cur, model.output_s)` → 当"verify 模式"开启时改用 `output_d2t`（`[hidden, 32768]` 的受限权重） |
| 2 | `src/llama-ext.h` | 新增 `llama_set_verify_d2t(ctx, ids, n, enable)`（staging API，和 `llama_set_embeddings_nextn` 同级） |
| 3 | `src/llama-context.cpp` | 存 flag + ids；首次启用时把 `model.output` 的 d2t 行 gather 成一份 CPU 侧张量（2B 128 MB / 4B 256 MB），缓存起来 |
| 4 | `common/speculative.cpp` | verify 的 `llama_decode(ctx_tgt, ...)` 前后 `set_verify_d2t(..., true/false)`；acceptance 逻辑改成直接在这 32k logits 上取 argmax（不再需要散射） |
| 5 | `tools/`（新） | 从 GGUF 的 `d2t` 张量导出 `d2t.bin`（32768 × int32），或用 EAGLE3 meta.json 里的 d2t |

**d2t 表的来源**：EAGLE3 草稿 GGUF 里已经有（`tools/weight_transfer` 之外，`export_eagle3_gguf.py` 写进去的
`d2t` 张量）。ngram 不需要 d2t（它的候选 token 直接来自上下文），所以这条优化只对**草稿模型**（EAGLE3/MTP）有效，
对 ngram 无效。

> **重要限制**：ngram 的候选不来自固定词表，所以"受限输出头"这条路**救不了 ngram**。
> ngram 想转正，只能靠提高接受率（现在 QA 上 39.3%、mean len 2.10，需要 >2.43 才打平），
> 或者减少每轮固定开销。而 EAGLE3 的 verify 可以吃这个优化。

## 验证方式

1. **正确性**：`tools/spec_bench` 的 json 集，接受率应与改动前**完全一致**
   （因为 ARGMAX 的取值集合没变）；2B stock PPL 也应不变（PPL 走非 verify 路径）。
2. **收益**：`qa` 集上 EAGLE3 的 out_tps 从 5.12 提到 ~10-14。
3. **回归**：四个质量门 + `check_k128.py` 布局不变（这个改动不碰权重文件）。

## 风险

* gather 出的 128/256 MB 张量要放在 CPU 后端缓冲，需走 `ggml_backend_alloc_ctx_tensors`；
* `inp_out_ids` 机制已经存在（`qwen35.cpp:719`），只需换矩阵，不需要动位置选择；
* 若 `output.weight` 是 tied（= token_embd），gather 时注意取的是 `model.output` 而不是 `token_embd`。

## 打平条件（这条公式解释了所有观测）

投机每多草稿 1 个 token，成本是 `1 × 输出头(≈79 ms)`，收益是"少跑一次 decode"(≈174 ms)，
只有在它被接受时才发生：

```
p × 174 > 79   =>   p > 0.45        （全词表 verify）
```

对照实测的 per-draft 接受率：

| 负载 | p | 判定 | 实测 |
|---|---:|---|---:|
| JSON（结构重复） | 0.88 | ✅ 赚 | +52.7% |
| chat | 0.55 | ✅ 微赚 | +3.7% |
| QA | 0.39 | ❌ 亏 | −2.4% |

**如果把 verify 的头降到 32k（每位置 10 ms）**：

```
p × 174 > 10   =>   p > 0.057
```

几乎任何接受率都赚 —— 这就是为什么这个改动对**草稿模型**（EAGLE3/MTP）是决定性的，
对 ngram 无效（ngram 需要全词表 argmax 才能判接受）。

## 现成的模板：CED 的加载器

`src/models/llama-ced.cpp:53 llama_ced_load(path, buft)` 就是"把外部 fp16 矩阵变成一个
可参与图计算的张量"的完整范例，d2t 受限权重照抄即可：

```cpp
// 115-140 行附近的做法
size_t arena = ...;
ggml_init_params ip = { arena, nullptr, /*no_alloc =*/ true };
ggml_context * ctx_raw = ggml_init(ip);
ggml_tensor * k = ggml_new_tensor_2d(ctx_raw, wq_type, n_embd, kv_dim);
// ... 填数据 ...
ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx_raw, buft);  // buft 由调用方给
```

对应到 d2t：

```cpp
// 上下文里缓存一份 [n_embd, 32768] 的同类型张量（2B 128 MB / 4B 256 MB）
ggml_tensor * out_d2t = ggml_new_tensor_2d(ctx_raw, model.output->type,
                                           n_embd, n_draft_vocab);
// 逐行从 model.output 拷：out_d2t 的第 j 列 = output.weight 的第 ids[j] 列
```

注意 `model.output` 的行是"输出维度"，在 ggml 里是 `ne[1]`；`ggml_new_tensor_2d(a, t, n_embd, n_rows)`
的第二个参数是 `ne[1]`（行数），所以是按 `ne[1]` 索引 gather。

`buft` 从哪来：`llama_ced_load_from_env(..., buft)` 是由 `qwen35.cpp:293` 的 `static const llama_ced_ctx ced`
初始化的，那里没有 buft 参数（走默认 CPU）。d2t 权重同理可以直接用 CPU 后端。

## 执行顺序（下一轮照做）

1. `llama-ext.h` + `llama-context.cpp`：`llama_set_verify_d2t(ctx, ids, n, enable)`
2. `libllama` 侧：首次调用时按上面模板 gather 出 `out_d2t` 并缓存
3. `qwen35.cpp:727`：verify 模式且 `out_d2t` 可用时，把 `model.output` 换掉
4. `common/speculative.cpp`：verify 的 decode 前后开关；接受判定直接用 32k logits
5. **验证门（必须全过才继续）**：
   - `json` 集接受率**逐位等于** 88.4%（改动前）
   - 2B stock PPL 仍是 17.1009（PPL 不走进 verify 路径）
   - 四个质量门不变
   - 然后才看 out_tps

## 实测：ngram 在本 fork 里没有可调参数空间

| 配置 | out_tps | acceptance | 结论 |
|---|---:|---|---|
| n=3 m=3（默认） | 4.79 | 39.3% | 唯一能产生草稿的组合 |
| n=3 m=1 | 4.93 | **null** | 不产生草稿 |
| n=3 m=2 | 4.72 | **null** | 不产生草稿 |
| n=4 m=3 | 4.91 | **null** | 不产生草稿 |
| n=5 m=3 | 4.91 | **null** | 不产生草稿 |

`acceptance = null` 表示 server 没有打印 draft acceptance 行，即一次草稿都没发。
⇒ **这个 ngram 实现写死了 3-gram + 扩展 3**，`size_n != 3` 或 `size_m < 3` 都直接不工作。
所以"调 n/m 提高接受率"这条路不存在，QA 负载上就是结构性 −2.4%。

## 环境变量版受限头：不可行（已验证）

先试了最省事的做法：`LLAMA_QWEN35_D2T=<ids>` 时**始终**用受限头。编译通过，但
**语义上行不通**：采样器和投机代码都按全词表索引 logits，受限头只给 32k 个值 ⇒ 越界。
已回滚（`git checkout qwen35.cpp` + 删除 `llama-d2t.h`），工作区干净。

⇒ 必须做成**仅 verify 生效的 per-decode 开关**（本文件开头列的第 1-4 步）。

## 唯一现在就能拿到的正收益：自适应开关

`p > 0.45` 才赚。实测 p 随负载变化很大：JSON 0.88、chat 0.55、QA 0.39。
所以在投机循环里统计最近 N 轮的接受率，低于阈值就禁用投机 =>
**QA 从 −2.4% 变回 ~0%，JSON 保持 +52.7%**，平均永远非负。

这是零风险的兜底方案，不依赖任何 C++ 结构改动（只动 common/speculative.cpp 的开关逻辑）。
