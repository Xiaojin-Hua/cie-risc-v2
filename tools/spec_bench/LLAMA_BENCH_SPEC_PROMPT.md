# 任务：给 llama-bench 增加投机解码（speculative decoding）支持

## 背景

- **仓库**：板端 `/home/bianbu/cie-risc-v2`（SpacemiT K1 RISC-V 开发板上的 llama.cpp fork）
- **连接**：`ssh -p 2222 root@localhost`；从 Mac 用 `./k1.sh '<cmd>'`、`./k1scp.sh <src> root@localhost:<dst>`
- **分支**：`dev`（`git log --oneline -3` 确认）
- **构建目录**：`llama.cpp/build-k1`（已存在，**不要重建**）

## 目标

让 `llama-bench` 能测投机解码，输出与普通 `tg128` 同口径的 t/s，
从而可以直接和基线（2B 约 5.2–5.35 t/s）对比，不必经过 llama-server 的 HTTP 开销。

## 现状（已确认）

`llama.cpp/tools/llama-bench/llama-bench.cpp`：

- 第 2193 行：prompt 阶段 `llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tokens))`
- 第 2214 行起：生成循环

```cpp
for (int i = 0; i < n_gen; i++) {
    int res = llama_decode(ctx, llama_batch_get_one(&token, 1));
    // 然后从 logits 采样下一个 token
}
```

## 需要做的改动

### 1. 新增 CLI 参数

参照 llama-bench 现有的 `add_opt` 写法：

| 参数 | 含义 |
|---|---|
| `-md, --spec-draft-model PATH` | 草稿模型 GGUF |
| `--spec-type TYPE` | 目前只需 `draft-eagle3`（可选透传 `ngram-simple`） |
| `--spec-draft-n-max N` | 每轮最多草稿几个 token（默认 3） |

### 2. 创建 common_speculative

API 在 `llama.cpp/common/speculative.h`：

```cpp
common_speculative_init(llama_context * ctx_tgt, llama_context * ctx_dft, ...);
common_speculative_get_draft_params(spec, seq_id);
common_speculative_draft(spec);                  // 产出草稿
common_speculative_accept(spec, seq_id, n_accepted);
```

### 3. 替换生成循环

改为 draft → verify → accept 循环。**参考实现在同一个仓库里**：

- `llama.cpp/tools/server/server-context.cpp`
- 搜索 `verify and try to accept the draft`（真正的 verify/accept 逻辑）
- 搜索 `common_speculative_accept(`（第 3912 行）
- 搜索 `draft acceptance = `（第 677 行，统计打印）

verify 的 batch 构造：一次 `llama_decode` 喂入 `[last_token, draft_0, ..., draft_{k-1}]`，
并且 **`batch.logits[i] = 1` 要对每个草稿位置都置 1**（每个草稿都要判接受）。

### 4. 计时口径（最容易做错的地方）

llama-bench 现在按**循环次数**算 token 数；投机每轮产出 **1~k+1** 个 token。
必须累加**实际产出的 token 数**再算 t/s，否则数字会被高估 k 倍。

建议同时打印 `mean_len`（实际 token 数 / 轮数）和接受率，方便与 server 对照。

## 已知的坑（都是实际踩过的）

1. **不要构建 `llama-server`**：它的 CMake 目标会去 HuggingFace 下载 Web UI
   （`dist.tar.gz`），板子无外网会卡死十几分钟。只构建 `llama-bench`：
   `cmake --build build-k1 -j8 --target llama-bench`
   （若改动了 `common/`，先 `--target llama-common`，再 `--target llama-bench`）

2. **编译器必须是 gcc-14/g++-14**：板子默认 gcc 13.2 缺少 `vfloat16m2_t`。
   若需重配：
   ```
   -DCMAKE_C_COMPILER=/usr/bin/gcc-14 -DCMAKE_CXX_COMPILER=/usr/bin/g++-14
   -DGGML_RV_ZBA=ON -DGGML_RV_ZFH=ON -DGGML_RV_ZVFH=ON -DGGML_RVV=ON
   -DGGML_CPU_RISCV64_SPACEMIT=ON
   ```

3. **ggml 后端分配**：`ggml_backend_alloc_ctx_tensors` 需要 `ggml_backend_t`；
   若手上是 buffer_type，必须用 `ggml_backend_alloc_ctx_tensors_from_buft`。

4. **板子有残余负载**：跑前 `cat /proc/loadavg`，理想 < 1.5；
   `code-server` 常驻会占核，必要时 `taskset -c 0-3` 绑核。

5. **跑之前 `pkill -9 -x llama-server`**，避免残留进程占内存/CPU。

## 验收标准（必须全部满足）

1. **基线不变**
   ```
   ./llama-bench -m <2B> -t 4 -p 0 -n 128 -fa 1 -r 2
   ```
   tg128 仍应在 **5.2–5.35 t/s**（改动前的值，不能漂移）。

2. **投机跑通**
   ```
   -md /home/bianbu/kwork/specdecode/e3n_q4.gguf --spec-type draft-eagle3 --spec-draft-n-max 3
   ```
   能跑完 128 token 并打印 mean_len。

3. **接受率与 server 一致**：同一草稿下，llama-bench 报的接受率应与
   llama-server 的 `draft acceptance = 0.88462 ... mean len = 3.63` **接近（±2% 内）**。
   差很多说明 verify 的 batch 构造或接受判定写错了。

4. **数字可解释**：len=3.63 时，每 token 时间约等于
   `(verify(4) + 草稿步) / 3.63`。当前实测 verify(4)≈422 ms、草稿步≈264 ms，
   预期约 **189 ms/token**，与基线 187 ms/token **基本持平**。
   如果测出 +20%，几乎一定是 token 计数算错了。

## 现有模型资产（板端）

| 文件 | 说明 |
|---|---|
| `/home/bianbu/kwork/nongemm/ssm_all_k128.gguf` | 2B 目标模型（Q4_0_K128） |
| `/home/bianbu/kwork/specdecode/e3n_q4.gguf` | 重训草稿，接受率 **88.5%** / mean_len 3.63 |
| `/home/bianbu/kwork/specdecode/e3g_allq4.gguf` | 原始草稿，接受率 79.5% / 3.34 |

## 交付

1. `git diff` 形式的补丁（只改 `tools/llama-bench/llama-bench.cpp` 和必要的 CMake）
2. 一张测量表：基线 / k=3 / k=5 的 t/s、mean_len、接受率
3. 明确说明与 server 口径的差异（如有）
4. **不要 commit**，也不要 `git reset --hard`，改动留在工作区让用户 review

## 参考：当前已测数字（server 口径，供对照）

| 配置 | evalscope t/s | server eval time (ms/token) | 接受率 | mean_len |
|---|---:|---:|---:|---:|
| 基线 | 4.75 | 187（llama-bench 值） | — | — |
| EAGLE3 k=3 | 4.86 | 192.75 | 88.5% | 3.63 |
| EAGLE3 k=5 | 3.37 | — | 45.4% | 3.26 |
