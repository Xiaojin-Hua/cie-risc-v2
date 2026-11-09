# spec_bench — 投机解码的标准测法

用 **evalscope perf**（统一的开源压测工具）在 `llama-server` 上做单流 A/B。
目标是让不同投机方案（ngram / MTP / EAGLE3）之间、以及和基线之间的数字可比。

```
tools/spec_bench/
├── run_bench.sh        单次测量：模型 × 投机配置 × prompt 集 → 一行 JSON
├── gen_prompts.py      生成 prompt 集（qa / json / summary）
├── report.py           汇总多次测量成表格
└── prompts/            qa.txt  json.txt  summary.txt
```

## 1. 为什么不能随便测

我们在 2026-10-24 用裸文本续写（wikitext 片段）测出过 ngram +63.6% / +86.6%，
看起来惊人，**但那是假的**：模型对无指令的裸片段会退化成复读循环

```
"The guns were also equipped with a 12 mm gun for the 12 mm gun for the 12 mm ..."
"Du Fu 's main reason for Du Fu 's main reason for Du Fu 's main reason ..."
```

而复读场景下 ngram 自然命中 89–97%。**这不是优化，是作弊。**

所以测法必须满足：

| 要求 | 原因 |
|---|---|
| prompt 是 **instruct 形式**（带任务指令） | 避免退化复读 |
| 必须记录 `draft acceptance` / `mean len` | 唯一能判断投机是否真生效的指标 |
| A/B **交替 + 3 次重复** | 板端 run-to-run 有 ±3% 波动 |
| 单流（`--parallel 1`） | 并发下投机是负收益（实测 p=2 −15%、p=4 −31%） |
| 输出**去重检查** | 唯一 n-gram 比例太低 ⇒ 数字无效，丢弃 |

## 2. 用法

```bash
# 单次测量（模型路径、投机 flags、prompt 集、输出 token 数）
./tools/spec_bench/run_bench.sh \
    --model /path/model.gguf \
    --spec "--spec-type ngram-simple --spec-ngram-simple-size-n 3 --spec-ngram-simple-size-m 3" \
    --prompts qa --max-tokens 128 --tag base_qa

# 不带投机就是基线（--spec ""）
./tools/spec_bench/run_bench.sh --model /path/model.gguf --spec "" \
    --prompts qa --max-tokens 128 --tag base_qa

# 汇总
python3 tools/spec_bench/report.py results/*.json
```

每次运行写一个 JSON：`{tag, model, spec, prompts, max_tokens, out_tps, ttft_ms, tpot_ms,
acceptance, mean_len, degenerate}`。

## 3. prompt 集

| 集 | 内容 | 预期投机收益 |
|---|---|---|
| `qa` | 中文问答（指令式，答案是中短句） | 低（+0~5%） |
| `json` | 结构化抽取（固定字段，多个对象） | **高**（结构重复） |
| `summary` | 给一段 wikitext，要求中文概括 | 低（自然语言） |

`gen_prompts.py` 从 wikitext 语料自动生成 `summary` 集（每条 = 指令 + 一段真实文本）。

**`json` 集上的收益是上界**：它刻意要求 15 个同构对象，结构重复度极高。
报告时必须同时给出 `qa`/`summary` 的数字，不能只报 `json`。

## 4. 已知的有效数字（2B / 4B，修好的构建，无 CED）

| 配置 | 负载 | baseline | ngram | 变化 | 接受率/len |
|---|---|---:|---:|---:|---:|
| 2B + ngram n3m3 | JSON | 5.14 | **7.85** | **+52.7%** | 88.4% / 3.65 |
| 2B + ngram n3m3 | chat | 5.15 | 5.34 | +3.7% | 55.6% / 1.25 |
| 4B + ngram n3m3 | JSON | 2.46 | **3.07** | **+24.8%** | 76.3% / 3.25 |
| 2B + EAGLE3 k=3 | JSON | 5.14 | 5.12 | −0.4% | 77.2% / 3.32 |

## 5. 坑

1. **evalscope 客户端统计**：`line_by_line` 数据集必须带 `--dataset-path`，不能带
   `--min-prompt-length`（否则 "Dataset is empty!"）；输出目录不可复用
   （`benchmark_data.db` 冲突）⇒ 每次 `rm -rf outputs/*`。
2. **thinking 模型**：请求必须带 `chat_template_kwargs.enable_thinking=false`，
   否则 token 全花在 `reasoning_content` 上、`content` 为空（准确率评测会得 0%）。
3. **CED 与投机同时开**：CED 只影响 prefill，理论上不影响 decode 收益，但
   **CED 与 EAGLE3 不兼容**（前者不为 split 之后的层产出 `t_layer_inp`）。
   测投机时统一 `env -u LLAMA_QWEN35_CED`，只在最后单独验证 CED+ngram 的组合。
4. **`--spec-draft-n-max` 只对草稿模型生效**；ngram 用 `--spec-ngram-simple-size-m` 控制最大草稿数。
5. **批形状病态**：本板 m=3（k=2）和 m=6（k=5）的 verify 特别慢，k=3（m=4）是好形状。
   扫 k 时必须重复测，单点不可信。

## 6. 已验证的正收益（2026-10-24）

ngram n3m3 + 自适应开关，evalscope perf，单流，128 token，QA prompt 集（8 条中文问答）：

| 模型 | baseline | ngram | 变化 | 接受率 / mean len | 输出退化 |
|---|---:|---:|---:|---:|---|
| 2B | 4.75 | 4.71 | −0.8% | 39% / 2.50 | 否 |
| **4B** | **2.29** | **2.46 / 2.49** | **+7.4% / +8.7%** | **88.9% / 3.67** | **否** |

4B 的两次 baseline 完全一致（2.29），两轮投机分别 2.46 / 2.49，收益可复现。

**为什么 4B 赚、2B 不赚**：打平条件是 `mean_len > 1 + m × head_ms/token_ms`，
而 head 在整次前向里的占比随模型变大而下降（2B 约 45%，4B 约 30%），
同时 4B 在这组中文问答上的接受率更高（88.9% vs 39%）。两者叠加，4B 跨过门槛而 2B 没有。

2B 上投机目前无正收益；要让 2B 也赚，必须降低 verify 的 head 成本（d2t 受限头，见
D2T_LOGITS_PLAN.md —— 散射方案已证伪，需走"不散射、直接消费 32k logits"的路线）。
