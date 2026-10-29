# K1 上 Qwen3.5-2B 的 per-op 耗时占比（打点结果汇总）

数据来自仓库里已有的 `[spxnode]` 逐节点打点（每行一个 op：`us=` 耗时、`ne=` 形状、
`src0=` 权重类型），用 `tools/op_share.py` 重新聚合。**所有数字都是 thread-0、
barrier-inclusive 口径，op-table 覆盖 wall time 的 ~98%**；打点本身有 5–30% 开销，
所以只看占比，不看绝对速度。

## 0. 数据来源

| 文件 | 配置 | 用途 |
|---|---|---|
| `results/2026-10-16-k1-snapshot/data/snapshot.log` | `ssm_all_k128` + **CED TAIL=128**（生产配置），pp128/512/2048/4096 | 主表（prefill） |
| `results/2026-10-15-k1-fa-budget/data/phase.log` | 同上，pp2048（独立复测） | 交叉验证 |
| `results/2026-10-13-k1-gdn/data/opstats_vreg.log` | `arm_k128_q8src` + **CED TAIL=16**，pp512 | TAIL=16 档位 |
| `results/2026-10-11-k1-prefill-non-gemm/data/analysis/*.txt` | pp512 + TAIL=16 | GEMM / 非 GEMM 分类、MUL_MAT 内部相位 |
| `.claude/memory/optimization-log.md`（2026-09-02 采样） | 旧模型（Q4_1 + q6_k LM head）tg32 | decode 占比（唯一一份） |

打点开关（当前 build 的 `libggml-cpu.so` 里都有）：

```bash
GGML_SPACEMIT_OP_TRACE_GRAPHS=2   # 前 2 个 prefill 图逐节点打印 [spxnode]
GGML_SPACEMIT_OP_STATS=1          # 退出时打印聚合
GGML_SPACEMIT_MM_PHASES=1         # MUL_MAT 内部分相（quant / barrier / gemm / TCM copy）
```

## 1. Prefill：各 op 占比（生产配置 CED TAIL=128）

`llama-bench -p N -n 0 -b N -ub N -t 4 -fa 1`，2B Q4_0_K128。

| op | pp128 | pp512 | pp2048 | pp4096 |
|---|---:|---:|---:|---:|
| **MUL_MAT** | **76.9%** | **72.6%** | **65.6%** | **58.8%** |
| FLASH_ATTN_EXT | 2.2% | 5.8% | 14.4% | **23.3%** |
| GATED_DELTA_NET | 11.3% | 10.9% | 9.9% | 8.8% |
| SSM_CONV_STATE | 2.6% | 2.6% | 2.4% | 2.1% |
| RMS_NORM | 2.2% | 2.6% | 2.5% | 2.3% |
| GLU | 2.0% | 2.0% | 1.8% | 1.6% |
| UNARY | 1.4% | 1.4% | 1.2% | 1.1% |
| ADD | 0.3% | 0.8% | 0.9% | 0.8% |
| SCALE / MUL / ROPE / CONT / GET_ROWS / SET_ROWS / CONCAT / CPY | 合计 ≈1.2% | ≈1.1% | ≈1.2% | ≈1.1% |
| **非 MUL_MAT 合计** | **23.1%** | **27.4%** | **34.4%** | **41.2%** |
| 每 token 总耗时（op-table） | 19.93 ms | 14.91 ms | 15.13 ms | 16.61 ms |

关键形状：**MUL_MAT 的每 token 成本几乎是常数**（15.3 → 10.8 → 9.9 → 9.8 ms/token，
n≥512 后进入算力饱和），而 **FLASH_ATTN 随 n 线性涨**（0.45 → 0.87 → 2.17 → 3.87 ms/token），
到 4K 时它已经吃掉 23% 的时间——这是长上下文 prefill 的第一优先级。

TAIL=16 档位（`opstats_vreg.log`，pp512）：MUL_MAT 69–71%、GATED_DELTA_NET 10–13%、
FA 6.6%、SSM_CONV 2.5%、RMS 2.6% —— 与 TAIL=128 同构，只是前缀更长、MUL_MAT 占比略降。

## 2. MUL_MAT 内部：形状分解（pp128）

| 形状 `ne` | 占总时间 | 占 MUL_MAT | 说明 |
|---|---:|---:|---|
| `[6144,128]` | 44.3% | 57.7% | attn_qkv / ffn_gate / ffn_up（18 层 × 3） |
| `[2048,128]` | 26.6% | 34.6% | attn_gate / ssm_out / ffn_down |
| `[4096,128]` | 2.9% | 3.7% | 其它投影 |
| `[248320,1]` | 1.9% | 2.5% | LM head（tied token_embd） |
| `[512,128]` / `[16,128]` | 1.1% | 1.5% | ssm 小投影 |

即 **FFN 的两组 6144 宽矩阵 + 4 组 2048 宽矩阵 ≈ MUL_MAT 的 92%**，全部走 IME1（Q4_0_K128）。

## 3. MUL_MAT 内部分相（pp512，TAIL=16，`2026-10-11` 分析）

| 相位 | 占 wall |
|---|---:|
| GEMM kernel（IME1 i8i4） | 59.26% |
| GEMM kernel（generic CPU 路径） | 6.15% |
| (a) post-GEMM C pass（GLU 1.53% + ADD 0.42%） | 1.95% |
| (b) layout/staging（TCM staging copy 1.19% + CONT/CONCAT/GET_ROWS） | 1.39% |
| (c) A quant/pack | 1.65% |
| (d) 独立算子（GDN/SSM_CONV/FA/RMS…） | 27.89% |

⇒ prefill 里"GEMM 本体"约 65% wall，"非 GEMM"约 33% wall；后者中
**GDN+SSM_CONV+FA ≈ 24%**，是 2B prefill 的第二战场。

## 4. Decode 占比（tg32，旧 build 采样）

| op | 占比 |
|---|---:|
| MUL_MAT | **84.1%** |
| GATED_DELTA_NET | 7.7% |
| GET_ROWS | 3.4% |
| CPY（state/embd 拷贝） | 3.4% |
| CONCAT | 0.8% |
| FLASH_ATTN_EXT | 0.2% |

> 这份是 2026-09-02 用 `GGML_SPINE_PERF=1`（thread0 口径，采样开销 ~5.5%）在旧模型上测的：
> 当时 `ssm_out` 还是 Q5_K（generic 路径，占 decode MUL_MAT 的 ~23%）。现在全 K128 走 IME1，
> MUL_MAT 的占比只会更高（≈85–90%），GDN 的绝对时间（~1.0 ms/层）不变。
> **当前 build 没有 decode 图打点**（`[spxnode]` 只在 `n_tokens>1` 的 prefill 图上打印），
> 要最新数字需要在空载板上跑一次 `-p 0 -n 32` + `GGML_SPACEMIT_OP_STATS=1`（约 10 s）。

## 5. 结论（优化优先级）

1. **prefill**：MUL_MAT 仍是第一大头（58–77%），但已全部 IME1 硬件；剩余可挖的是
   **FA 的 O(n) 增长**（4K 时 23%）、**GDN ~10%**、**SSM_CONV 2.5%**。
2. **decode**：MUL_MAT ≈85%（DDR 权重流），非算力受限；`GET_ROWS+CPY` 6.8% 是
   state/embedding 拷贝，属"可省但要改数据流"的部分。
3. n≥512 后 MUL_MAT 每 token 成本基本不变（10.8→9.8 ms），非 MUL_MAT 从 4.1→6.8 ms/token
   ⇒ 同一份打点的总吞吐 pp512 67.1 → pp2048 66.1 → pp4096 60.2 t/s（**−10%**），
   掉的全是 FA。
