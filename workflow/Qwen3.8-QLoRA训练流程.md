# Qwen3.8-27B 猫娘 QLoRA 标准流程

这是参考机跑通 v1 和 v1.2 的流程。公开仓库只带这份流程、同目录的 LlamaFactory 补丁，以及 `corpus/tool-800.jsonl`。NekoCOT 和 NekoMath 不放进仓库，从下面的链接自己下。盘符是参考机的，换机器就改。走不通的地方由 agent 改文档和脚本，目标是走完两程。

路径按参考机写。步骤不依赖别的笔记。

语料三份都在下面「语料」一节。第一程一份。第二程的数学和 tool call 拆开给。流程从只读 BF16 底座开始：第一步转成 NF4，再注册数据，训到 `F:\models` 里的 Q4_K_M。

猫娘这条线分两程。第一程从干净 NF4 训出 **v1**。第二程接着 v1 的终档 LoRA，不重放那 3 万条，训出 **v1.2**。两程都走下面同一套停续、NaN、抽测和 merge。超参差在第 3 节：第一程是「新 LoRA」，第二程是「接着一份已经训完的 LoRA」。

机器：R9700 32 GB / 64 GB RAM / Windows 11 / WSL2 训练 / Windows `E:\llama.cpp.latest` 量化。GPU 重活只白天。WSL 内存约 47 GB，27B BF16 不能整模进 RAM。

| 用途 | Windows | WSL |
|---|---|---|
| 训练根（venv、NF4、LoRA、日志、脚本） | `G:\neko-sft` | `/mnt/g/neko-sft` |
| 只读 BF16 底座 | `G:\Qwen3.8-27B-Uncensored` | `/mnt/g/Qwen3.8-27B-Uncensored` |
| llama.cpp | `E:\llama.cpp.latest` | `/mnt/e/llama.cpp.latest` |
| 最终 GGUF | `F:\models\<Name>\` | `/mnt/f/models/<Name>/` |
| llama-server 配置 | `F:\models\models.ini` | — |

底座下载：https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored

工作根一律 `cd /mnt/g/neko-sft`。**不要往 `G:\Qwen3.8-27B-Uncensored` 写任何产物。**

磁盘按一次训练留空：

| 东西 | 大约 |
|---|---|
| 只读 BF16 底座 | 55 GB，用户自己下载 |
| NF4 | 15 GB，第 1 步从这份 BF16 转出 |
| 分片 merge 的 HF BF16 | 52 GB，留下 |
| 中间 BF16 GGUF | 51 GB，量化完删 |
| Q4_K_M | 16 GB |

空机器按这个顺序，不要从「语料」的打包命令倒着做：

1. 建 `outputs/logs`、`data`、`pack/v1.2`。命令在「前提」里，要在安装脚本的日志重定向之前。脚本在训练根的 `scripts/`，参考机是 `G:\neko-sft\scripts`。补丁是本仓库 [`llamafactory-qwen3.8-neko.patch`](llamafactory-qwen3.8-neko.patch)。安装脚本若写死别的路径，改成这份。
2. 前提：WSL、`/opt/rocm`、`python3.12`、`git`、`curl`、`tmux`，然后两个安装脚本。
3. 用户自己下载 BF16。
4. 第 1 步转 NF4。
5. 把三份语料复制到位，写 `dataset_info.json`。
6. venv 装完之后再跑 `pack_v1_2.py`。
7. YAML 从第 4 节抄。不要把已经跑过的三份活 YAML 当新版本的底稿。

---

## 两程

### 第一程：v1

从干净 NF4 起，不接任何 adapter。数据是原样 [NekoCOT-30K](https://huggingface.co/datasets/liumindmind/NekoCOT-30K)，30301 条，alpaca，cutoff 1024。怎么下到盘上见「语料」。3788 step，学习率 `1e-4`，warmup 100。1 epoch 已收工。

| | |
|---|---|
| LoRA | `G:\neko-sft\outputs\lora\v1-dumb`（与 `checkpoint-3788` 同一份） |
| Q4_K_M | `F:\models\Qwen3.8-27B-v1-dumb\Qwen3.8-27B-v1-dumb-Q4_K_M.gguf`，16.0 GB |
| 预设 | `models.ini` 的 `[qwen3.8-27b-neko-v1.0]` |

猫娘角色已经融入。空 system 问「你是谁」是猫娘（自称小猫咪、叫主人、括号动作），不再是通义千问。春日影、寅虎装傻，按这一程的判据算过。Q4 上这次没有 LoRA 抽测里的「蹭」循环。

短板有两处，所以才有第二程：

- **tool call 不会发。** 原样 NekoCOT 没有 function call。问今天的日期，或者点名 `web_search_exa` / `web_fetch_exa`，它把搜索演成动作，再编一个日子。请求里的工具列表是发出去的，模型不走这个通道。
- **数学是短板。** 这一程的目标句是装不懂，没有真推导。不要指望再训一 epoch 就长出计算。

### 第二程：v1.2

接着 v1 终档，只加载 LoRA 权重。开训时不 `resume_from_checkpoint`，v1 的 cosine 已经走完。不重放 30301 条，不放 v2 改写、身份锚、NSFW。原料是两份拆开的文件：数学 2303 和 tool call 800，不要先合成一份再当来源。训练前用 `scripts/pack_v1_2.py` 打成 3103 条 sharegpt，cutoff 2048。388 step，学习率 `5e-5`，warmup 20，新 cosine。春日影、寅虎继续装傻，不算失败。

2026-09-25 11:26 收工，结果可以接受。中间停过一次：当晚停在 `checkpoint-200`，次日同一次跑续上，不是新开一份 cosine。step 201 和 252 各 skip 一次，都是 consecutive=1。末段 loss 约 0.53–0.58。日志汇总 `train_loss 0.2811` 是续训把损失除以全程 388 步，不要当末段。cutoff 2048 大约 36–45 秒/步。空 system 四题过了再 merge：你是谁是猫娘；`37²−18²=1045`；日期打出 `get_datetime`；春日影编成春天的歌，不挡。原文 `outputs/logs/v1.2-math-tool-compare.md`。

| | |
|---|---|
| 起点 | `outputs/lora/v1-dumb` |
| 数学原料 | [NekoMath](https://huggingface.co/datasets/liumindmind/NekoMath)，2303 条。不随本仓库分发 |
| tool 原料 | 本仓库 [`corpus/tool-800.jsonl`](../corpus/tool-800.jsonl)，800 条。来源 ember.cc |
| 训练文件 | `data/v1.2-math-tool-sharegpt.json`，注册名 `v1_2_math_tool`。由上面两份打出来，不是第三份来源 |
| LoRA | `outputs/lora/v1.2-math-tool`。不要写回 `v1-dumb` |
| 已落地 | merge `outputs/merged/v1.2-math-tool-bf16`（18 片留下）。Q4 `F:\models\Qwen3.8-27B-v1.2-NekoMath\Qwen3.8-27B-v1.2-NekoMath-Q4_K_M.gguf`，16021 MiB，4.92 BPW。中间 BF16 GGUF 已删。预设 `[qwen3.8-27b-neko-v1.2]`。OpenClaw 加了同名别名，`primary` 没改 |

不过的线：你是谁说成千问，或问日期不调 `get_datetime`。不过就不 merge。

---

## 语料

三份原料。第一程一份。第二程两份：数学和 tool call 分开给，打成一份 sharegpt 只发生在开训之前。

### 第一程：NekoCOT-30K

来源：https://huggingface.co/datasets/liumindmind/NekoCOT-30K

中文猫娘 CoT，30301 条。字段是 `instruction`、`output`、`type`。`output` 以 `<think>` 开头。原样用，不改写，没有 tool。约 89 MB。

不随本仓库分发。从上面的链接下载，得到 30301 条、字段为 `instruction` / `output` / `type` 的 JSON，开训前放到 `data/NekoCOT-30K.json`。复制结果不能是符号链接。若目标已是符号链接，先删链接再复制。

### 第二程数学：NekoMath

来源：https://huggingface.co/datasets/liumindmind/NekoMath

2303 条中文猫娘数学。上游文件是 `data/train.jsonl`，字段是 `problem_zh`、`cot`、`response`、`final_answer`。卡片上的助手输出是 `<think>\n{cot}\n</think>\n\n{response}`，`problem_zh` 当用户输入。生成侧写的是 Claude Opus-4.8、Claude Sonnet-5、GPT-5.6-sol。范围是数论、几何、代数、组合、概率与统计、微积分、线性代数。

不随本仓库分发，和 tool call 不在同一个文件里。从上面的链接取 `data/train.jsonl`（2303 行）。开训前包成 jsonl：`instruction` 是 `problem_zh`，`output` 以 `<think>` 开头、`</think>` 后接回复，并保留 `final_answer`。放到 `pack/v1.2/nekomath-2303.jsonl`。参考机上 `</think>` 后面是一个换行。卡片示例是 `<think>\n{cot}\n</think>\n\n{response}`。

### 第二程 tool call

来源：ember.cc。800 条，随本仓库给出，和数学不在同一个文件里：

[`corpus/tool-800.jsonl`](../corpus/tool-800.jsonl)（800 行）

开训前复制到训练根的 `pack/v1.2/tool-800.jsonl`。

每行是 sharegpt。`conversations` 的角色是 `human`、`function_call`、`observation`、`gpt`。`function_call` 和 `gpt` 带非空 think。`tools` 是非空 JSON 字符串。工具名是 `get_datetime`、`get_info`、`web_search_exa`、`web_fetch_exa`。`model` 是 `grok-4.7`，腔是猫娘。调用和 observation 保持轨迹原文。

| kind | 条数 |
|---|---|
| search | 200 |
| fetch | 120 |
| datetime | 80 |
| search_fetch | 160 |
| get_info | 40 |
| negative | 120 |
| wrong_tool | 80 |

NekoCOT 下到 `data/NekoCOT-30K.json`，NekoMath 包好后放到 `pack/v1.2/nekomath-2303.jsonl`。本仓库只额外复制 tool 文件。复制不需要 venv。若 `data/NekoCOT-30K.json` 已是符号链接，先删链接再放下载结果。仓库路径按克隆位置改，下面是参考机把本仓库放在 `G:\qwen3.8-agentic-neko` 时复制 tool 的命令：

```bash
MSYS_NO_PATHCONV=1 wsl bash -lc 'cp /mnt/g/qwen3.8-agentic-neko/corpus/tool-800.jsonl /mnt/g/neko-sft/pack/v1.2/tool-800.jsonl'
```

空的 `data/dataset_info.json` 写成：

```json
{
  "nekocot_30k": {
    "file_name": "NekoCOT-30K.json",
    "formatting": "alpaca",
    "columns": { "prompt": "instruction", "response": "output" }
  },
  "v1_2_math_tool": {
    "file_name": "v1.2-math-tool-sharegpt.json",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations", "tools": "tools" }
  }
}
```

盘上若已有别的键，只追加这两个，不要整文件覆盖。打包要等前提里的 venv 装完。

### 打成训练文件

两份都在 `G:\neko-sft\pack\v1.2\` 之后，在 WSL 里：

```bash
cd /mnt/g/neko-sft
.venv/bin/python scripts/pack_v1_2.py
```

`scripts/pack_v1_2.py` 不依赖别的目录。数学行的 `tools` 写成空字符串，tool 行保留自己的 `tools`。种子 `20260923` 打乱后写出 `pack/v1.2/v1.2-math-tool-sharegpt.json`，再复制到 `data/v1.2-math-tool-sharegpt.json`。复制结果不能是符号链接。条数必须是 3103。注册名 `v1_2_math_tool`。

---

## 精度

这是 QLoRA，不是整模 BF16 微调。YAML 开 `bf16: true`，不开 `pure_bf16`。

| 块 | 精度 |
|---|---|
| 底座线性层 | bitsandbytes **NF4** + double quant，冻结。`G:\neko-sft\models\nf4` |
| 前向 / 反向 | **BF16**。NF4 先反量化，矩阵乘走 `bnb_4bit_compute_dtype=bfloat16`。日志应有 `compute dtype: torch.bfloat16` |
| LoRA | 权重存 **FP32**。量化训练且非 `pure_bf16` 时，LLaMA-Factory 会 upcast 可训练参数 |
| LayerNorm | **FP32**。`upcast_layernorm: true` |
| 优化器状态 | **FP32**。`adamw_torch`，状态跟可训练参数 |
| `lm_head` | 第 1 步转 NF4 时跳过，训练时保持 **BF16** |

模型 config 里的 `mamba_ssm_dtype: float32` 是这份 Qwen3.8 自己的（text config 的 `model_type` 是 `qwen3_5_text`，线性注意力沿用这套字段），不是这套 YAML 另开的精度。

训练用的 NF4 和导出的 llama.cpp **Q4_K_M** 不是同一种量化。抽测过了再 merge：LoRA 合进只读 BF16，再转 Q4_K_M。

---

## 前提

环境和模型分开。环境是 venv 和补丁。模型的原始输入是 BF16 权重，不是 NF4，也不是 GGUF。

### 环境

干净环境按这个顺序做，不要跳，也不要自己 `pip install` 成「>=」。

前提（这两个脚本不管）：

- WSL2 Ubuntu。`%USERPROFILE%\.wslconfig`：`networkingMode=Mirrored`，`memory=51539607552`（48 GiB）。改完 `wsl --shutdown` 再开。
- WSL 里 `/opt/rocm` 要事先有，HIP `7.2.26015`。`python3.12` 是 **3.12.3**。`install-torch.sh` 会把 torch 的 `libhsa-runtime64.so` 链到 `/opt/rocm/lib/libhsa-runtime64.so.1`，这个文件不在就停。两个安装脚本不负责装 ROCm。读者自备与轮子匹配的 ROCm 7.2。`e:\_doc_\llama.cpp.guide\rocm-r9700-wsl-llamacpp-flow.md` 写的是 ROCm 7.2.4 / ROCDXG，版本对不上，不能当成这一步。
- WSL 里要有 `git`、`curl`、`tmux`。没有 `tmux`，第 1 步和开训的 `tmux new-session` 会失败。

然后：

```bash
# Git Bash。已经在 WSL 里就去掉前面的 MSYS_NO_PATHCONV=1 wsl bash -lc，直接跑引号里的命令。
# 日志目录要先在。install-torch.sh 不会建 outputs/logs，重定向会在脚本运行前失败。
MSYS_NO_PATHCONV=1 wsl bash -lc 'mkdir -p /mnt/g/neko-sft/outputs/logs /mnt/g/neko-sft/data /mnt/g/neko-sft/pack/v1.2'
MSYS_NO_PATHCONV=1 wsl bash -lc 'python3.12 -m venv /mnt/g/neko-sft/.venv'
MSYS_NO_PATHCONV=1 wsl bash -lc 'bash /mnt/g/neko-sft/scripts/install-torch.sh > /mnt/g/neko-sft/outputs/logs/install-torch.log 2>&1'
MSYS_NO_PATHCONV=1 wsl bash -lc 'bash /mnt/g/neko-sft/scripts/install-train-stack.sh > /mnt/g/neko-sft/outputs/logs/install-train-stack.log 2>&1'
```

`install-torch.sh` 从 `https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2` 下载四只轮子到 `~/neko-wheels`，装进 venv，并把 torch 里的 `libhsa-runtime64.so` 链到 `/opt/rocm`。venv 不存在会直接停。

`install-train-stack.sh` 必须在 torch 之后跑。包版本全部是 `==`，和下面这张表一致。跑完日志要有 `VERSIONS_OK` 和 `NF4_OK`。它会把 LlamaFactory 停在 `100e9a42c6c09f8f7849b70d60f3da445fb2024b`，再打 [`llamafactory-qwen3.8-neko.patch`](llamafactory-qwen3.8-neko.patch)。树里已经有 `SkipNonFiniteCallback` 就跳过。不要 `git pull`。

venv：`/mnt/g/neko-sft/.venv`。

| 包 | 版本 |
|---|---|
| pip | `26.2.1` |
| wheel | `0.48.0` |
| setuptools | `84.0.0` |
| ninja | `1.13.2` |
| torch | `2.9.1+rocm7.2.0.lw.git7e1940d4` |
| torchvision | `0.24.0+rocm7.2.0.gitb919bd0c` |
| torchaudio | `2.9.0+rocm7.2.0.gite3c6ee2b` |
| triton | `3.5.1+rocm7.2.0.gita272dfa8`（ROCm 轮，不是 PyPI 的 `triton==3.5.1`） |
| transformers | `5.17.0` |
| peft | `0.20.0` |
| trl | `0.24.0`（不要升） |
| accelerate | `1.15.0` |
| datasets | `5.0.1` |
| bitsandbytes | `0.50.2` |
| flash-linear-attention | `0.5.2` |
| fla-core | `0.5.2` |
| causal-conv1d | `1.7.0` |
| llamafactory | `0.9.6.dev0`（git `100e9a4`，2026-09-09，再加本地补丁） |
| huggingface_hub | `1.31.0` |
| tokenizers | `0.23.2` |
| safetensors | `0.8.0` |
| sentencepiece | `0.2.2` |
| protobuf | `7.36.1` |
| einops | `0.8.2` |
| PyYAML | `6.0.3` |
| tqdm | `4.70.1` |
| fire | `0.7.1` |
| omegaconf | `2.3.1` |
| pydantic | `2.13.5` |
| pandas | `3.0.5` |
| scipy | `1.18.1` |
| matplotlib | `3.11.2` |
| tiktoken | `0.14.0` |
| modelscope | `1.40.0` |
| hf-transfer | `0.1.9` |
| tyro | `1.0.16` |
| av | `18.1.0` |
| torchdata | `0.11.0` |
| numpy | `2.5.3` |
| packaging | `26.3` |

钉死的约束：

- torch / torchvision / torchaudio / triton 用本机 ROCm 轮（`~/neko-wheels`）。pip 换掉 torch 就坏了。装别的包之后要确认 `torch.__version__` 仍带 `rocm`。不要 `pip install triton==3.5.1`。
- `trl==0.24.0`。不要升。
- bitsandbytes `0.50.2`，`--no-deps`。bnb 只做 NF4 权重量化。不要 `paged_adamw_*` / `adamw_bnb_8bit`。
- `flash-linear-attention==0.5.2` 和 `fla-core==0.5.2` 用 `--no-deps`。**不要** `flash-linear-attention[rocm]`，那个 extra 会换掉 ROCm triton。
- causal-conv1d `1.7.0` 源码装，`--no-build-isolation`，`HIP_ARCHITECTURES=gfx1201`。装不上就停，不要改去装 CUDA torch。
- LlamaFactory 不用 PyPI。PyPI 的 0.9.5 把 transformers 钉在 `<=5.6`，带不起 Qwen3.8。脚本克隆 `https://github.com/hiyouga/LlamaFactory.git`，checkout `100e9a4`（2026-09-09，`[KT] Support Kimi K2.5/2.6 LoRA fine-tuning (#10826)`），`pip install --no-deps -e`。这一档 `src/llamafactory/extras/env.py` 的版本号是 `0.9.6.dev0`。上游 `pyproject.toml` 仍写 `transformers<=5.8`、`peft<=0.18.1`、`datasets<=4.0`；本机是 5.17.0 / 0.20.0 / 5.0.1，所以必须 `DISABLE_VERSION_CHECK=1`。

每次训练、以及第 1 步转 NF4，先 source `scripts/train-env.sh`：

| 变量 | 值 |
|---|---|
| `HSA_ENABLE_DXG_DETECTION` | `1` |
| `HIP_VISIBLE_DEVICES` | `0` |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE` | `TRUE` |
| `DISABLE_VERSION_CHECK` | `1`（上游钉 transformers<=5.8、peft<=0.18.1、datasets<=4.0；本机是 5.17.0 / 0.20.0 / 5.0.1） |
| `LLAMAFACTORY_ALLOW_TORCH29_CONV3D` | `1`（torch 2.9 的 Conv3d，vision 冻着，只放行） |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | 训练和转 NF4 的脚本里再设为 `1` |

`scripts/train-env.sh` 默认 `HF_ENDPOINT=https://hf-mirror.com`，只影响 source 之后的进程。token 留在用户环境，不写进脚本，不写进本文。

本地 LlamaFactory 补丁不要往上游提。基线是 `100e9a4`。完整 diff 在 [`llamafactory-qwen3.8-neko.patch`](llamafactory-qwen3.8-neko.patch)。`install-train-stack.sh` 负责 `git apply`，不要手打第二遍。本机这份树已经打过。

五处改动：

**`data/formatter.py`。** `qwen3_5` 模板的 think 结束标记是 `\n</think>\n\n`。语料里常常只写 `</think>`。原正则把整段结束标记当字面量，对不上就把整段内容当 JSON 解析，然后失败。补丁先把开标签和结束标记 `.strip()`，再匹配；切出来的 JSON 再 `.strip()`。这样 `function_call` 槽里「`<think>` + JSON」能过。

**`model/loader.py`，Conv3d。** torch 2.9 加上模型里的 Conv3d，上游直接 `ValueError`（[pytorch#166122](https://github.com/pytorch/pytorch/issues/166122)）。`LLAMAFACTORY_ALLOW_TORCH29_CONV3D=1` 时改成 warning，继续加载。vision 冻着，文本 SFT 不跑这层 Conv3d。没设这个变量仍然抛错。

**`model/loader.py`，`lm_head`。** 有的 PEFT / Trainer 路径会把打包的 NF4 `lm_head` 变成 `nn.Linear` 套 `Params4bit`。`F.linear` 看到的 shape 是 `1 × (vocab×hidden/2)`，前向直接坏。`_dequantize_4bit_lm_head` 用 `get_base_model()` 拿到内层模块，把 `lm_head` 反量化成冻结的 BF16 `Linear`。写在 `PeftModel` 自己身上无效。27B 大约多 2.5 GB VRAM。第 1 步转 NF4 时 `modules_to_not_convert=["lm_head"]` 是同一件事的前半；这份是加载时的保险。日志应有 `Dequantized inner lm_head from NF4 to bf16.`

**`model/model_utils/checkpointing.py`。** 这里换掉了 `gradient_checkpointing_enable`。transformers 5.17 调用它时多传 `every_n_layers` 和 `offload`，签名不接就会 `TypeError`。补丁把这两个参数接住，**没有**按 `every_n_layers` 跳层，也没有做 offload。checkpoint 仍是每一层。

**`train/callbacks.py` 和 `train/tuner.py`。** 加上 `SkipNonFiniteCallback`，并在 `_training_function` 里无条件挂上。`optimizer.step` 前看梯度；非有限就跳过这一 step，不写坏权重。连续 8 次才停，次数用 `NEKO_NAN_MAX_SKIP` 改。行为见第 9 节。开训日志应有 `SkipNonFiniteCallback on`。

### 模型

原始输入是只读 BF16：

| | |
|---|---|
| 路径 | `G:\Qwen3.8-27B-Uncensored`（WSL：`/mnt/g/Qwen3.8-27B-Uncensored`） |
| 下载 | https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored |
| 精度 | BF16，约 55 GB |
| 架构 | `Qwen3_5ForConditionalGeneration` |

这个目录只读。转换、LoRA、merge、GGUF 都写到别的地方。训练不直接加载这份 BF16：WSL 内存设的是 48 GiB，整模进 RAM 会爆。第 1 步把它转成 NF4，后面的 QLoRA 加载 NF4。抽测过了再 merge 时，才回到这份 BF16 上合 LoRA。

目录不在才下载。这时不要设 `HF_HUB_OFFLINE`。许可证是 Apache 2.0。页面若要求接受许可再点。只有页面要求时才把 token 放进环境变量 `HF_TOKEN`。这条 `snapshot_download` 没有 source `train-env.sh`，走的是当时环境里的 Hub。直连失败时，在命令前自行 `export HF_ENDPOINT`。

```bash
MSYS_NO_PATHCONV=1 wsl bash -lc '/mnt/g/neko-sft/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download(\"orcarouter/Qwen3.8-27B-Uncensored\", local_dir=\"/mnt/g/Qwen3.8-27B-Uncensored\")"'
```

不要用 GGUF、别人的 Q4、或冒烟 adapter 当起点。

---

## 1. BF16 → NF4

这是第一步。输入是上一节的 BF16，输出是 `G:\neko-sft\models\nf4`。QLoRA 加载的是这个目录，不是 BF16。

本机已经转好（2026-09-12）：约 15 GB、8 个 safetensors，`config.json` 里有 `quantization_config`。目录还在就不要重跑。脚本不会拒绝已有目录，要重转先把旧的 `models/nf4` 挪走。

开跑前 `:8001` 和 `:8188` 要空，这一步占 GPU。

```bash
MSYS_NO_PATHCONV=1 wsl bash -lc 'tmux new-session -d -s neko-nf4 "bash /mnt/g/neko-sft/scripts/convert-nf4.sh > /mnt/g/neko-sft/outputs/logs/convert-nf4.log 2>&1"'
```

`convert-nf4.sh` 会 source `train-env.sh`，设 `HF_HUB_OFFLINE=1`，RSS 看门狗默认 36 GB（`NF4_RSS_LIMIT_GB`），然后跑 `scripts/convert_nf4.py`。

不要 `from_pretrained(load_in_4bit=True)` 整模做完。18 个 BF16 shard 会一起 mmap，CPU 上还留着 BF16 副本。上次这样加载，大约 15% 权重时 RSS 已经 ~15 GB，外推会撞 36 GB 看门狗。脚本改成一次只开一个 shard：

1. `BitsAndBytesConfig`：NF4、double quant、`bnb_4bit_compute_dtype=bfloat16`。
2. `init_empty_weights` + `Qwen3_5ForConditionalGeneration._from_config`。transformers 5.17 没有公开的 `from_config`。
3. `replace_with_bnb_linear`，`modules_to_not_convert=["lm_head"]`。本机是 607 个 `Linear4bit`。`lm_head` 留在 BF16。
4. 读 `model.safetensors.index.json`，一次打开一个 shard。`set_module_tensor_to_device(..., "cuda:0")` 边搬边量化。跳过 `mtp.*`，以及 meta 模型里不存在的 key。每个 shard 结束 `gc.collect()`，RSS 回到约 1 GB。
5. 还有参数留在 meta 设备上就失败退出。checkpoint 里没有、只是运行时的 meta buffer 丢掉。
6. `save_pretrained(..., max_shard_size="2GB")`。再把 tokenizer 和 `generation_config.json`、`preprocessor_config.json` 等从 BF16 目录拷过来。
7. `from_pretrained` 回载这份 NF4，确认 `is_loaded_in_4bit=True`。

成功的最后一行是 `NF4_OK`。本机实测：量化 636 秒，写出 83 秒，转换峰值 RSS **4.72 GB**。回载 RSS 峰值约 16 GB，VRAM 约 15 GB。

训练日志里若出现 `quantization_bit will not affect on the PTQ-quantized models`，是预期：YAML 里的 `quantization_bit: 4` 不会对这份已经量化的权重再量化一次。

---

## 2. 注册数据

文件**复制**进 `data/`。不要软链到还在改的 pack 目录。新数据集用新名字，不要复用 `nekocot_30k`。

`tokenized_path` 每个版本独立，例如 `data/tokenized-<name>`。`overwrite_cache: false`。不要读上一版的 cache。

`cutoff_len` 会截断 tokenize 之后的尾巴。中文大约 1 token / 字。pack 时留着长样本，不要在这里删。默认 cutoff **2048**；OOM 再退 1024。v1 用 1024 是当时语料短，不是新版本的默认。

### Alpaca

v1 这种没有 tool、一条 instruction 对一条 output：

```json
"nekocot_30k": {
  "file_name": "NekoCOT-30K.json",
  "formatting": "alpaca",
  "columns": { "prompt": "instruction", "response": "output" }
}
```

`output` 以 `<think>` 开头，`</think>` 后还有正文。think 和正文都不能空。Alpaca 没有 `gpt` 这个角色。

第一程注册 `nekocot_30k`，文件 `data/NekoCOT-30K.json`。第二程注册 `v1_2_math_tool`，文件 `data/v1.2-math-tool-sharegpt.json`。这两份怎么到盘上，见「语料」。

### ShareGPT

带 tool、或多轮的包用这个。`dataset_info.json`：

```json
"example_pack": {
  "file_name": "example-sharegpt.json",
  "formatting": "sharegpt",
  "columns": { "messages": "conversations", "tools": "tools" }
}
```

一条记录：

```json
{
  "conversations": [
    {"from": "human", "value": "用户的话"},
    {"from": "function_call", "value": "<think>\n...\n</think>\n{\"name\": \"...\", \"arguments\": {}}"},
    {"from": "observation", "value": "工具返回"},
    {"from": "gpt", "value": "<think>\n...\n</think>\n可见回复"}
  ],
  "tools": ""
}
```

约束（pack 时就要过，不要等训练报错）：

- 第一轮是 `human`，最后一轮是 `gpt`
- 角色只允许 `human` / `gpt` / `function_call` / `observation`
- `gpt` 和 `function_call` 都必须有 `<think>...</think>`，think 和后面的正文都不空
- 正文里不要手写 `<tool_call>` XML。调用是 `function_call` 槽里的 JSON，模板负责套 XML
- `tools` 是字符串。没有工具的样本用 `""`，不要省掉这个字段，也不要写成 `null`
- 有工具的样本，`tools` 是工具定义的 JSON 字符串，不能是空的

---

## 3. 三种开法

先分清是哪一种。抄错会把 cosine 走完的优化器接着空转，或者把上一版目录写坏。第一程 v1 用「新 LoRA」。第二程 v1.2 用「接着一份已经训完的 LoRA」。同一次跑没训完、第二天再开，才用第三列。

| | 新 LoRA | 接着一份已经训完的 LoRA | 同一次跑，第二天续 |
|---|---|---|---|
| 什么时候 | 从干净 NF4 起 | 上一版 1 epoch 已收工，只加一小包新数据 | 昨晚停在某个 `checkpoint-N` |
| `adapter_name_or_path` | 不写 | 上一版**终档**目录 | 不写。权重在 checkpoint 里 |
| `resume_from_checkpoint` | 不写 | **不写** | 最后一份完好 checkpoint |
| `output_dir` | 新目录 | **另一个**新目录。不要写回上一版 | 还是这次的目录 |
| `learning_rate` | `1.0e-4` | `5.0e-5` | 不改。resume 时改 YAML 不会生效 |
| `warmup_steps` | `100` | `20` | 不改，scheduler 从 checkpoint 还原 |
| 优化器 | 新的 | 丢掉旧的，新 cosine | 必须有 `optimizer.pt` |

终档指 `output_dir` 根上那份 `adapter_config.json` + `adapter_model.safetensors`，不是某个中途 `checkpoint-*`，除非故意要那一档。

`v1-dumb-qlora.yaml`、`v1.2-math-tool-qlora.yaml`、`v2-smart-sfw-qlora.yaml` 是已经跑过的三程，里面的路径和数据名写死了。`resume_from_checkpoint` 已从这三份里删掉。新版本仍用下面这一份，再改标成 `scripts/<name>-qlora.yaml`。

---

## 4. YAML

```yaml
### model
model_name_or_path: /mnt/g/neko-sft/models/nf4
trust_remote_code: true
quantization_bit: 4
quantization_method: bnb
upcast_layernorm: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all
freeze_vision_tower: true
freeze_multi_modal_projector: true

### dataset
dataset: <dataset_info 里的名字>
dataset_dir: /mnt/g/neko-sft/data
template: qwen3_5
enable_thinking: true
cutoff_len: 2048
preprocessing_num_workers: 4
dataloader_num_workers: 0
overwrite_cache: false
tokenized_path: /mnt/g/neko-sft/data/tokenized-<name>

### output
output_dir: /mnt/g/neko-sft/outputs/lora/<name>
logging_steps: 5
save_steps: 100
save_total_limit: 8
plot_loss: true
overwrite_output_dir: false
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_steps: 100
bf16: true
optim: adamw_torch
max_grad_norm: 1.0
logging_nan_inf_filter: false
gradient_checkpointing: true
ddp_timeout: 180000000
```

接着上一版时才加 `adapter_name_or_path`，并把 `learning_rate` 改成 `5.0e-5`、`warmup_steps` 改成 `20`。同一次续跑才加 `resume_from_checkpoint`。两行不要同时出现。

上面这份 `cutoff_len: 2048` 是第二程和新开的默认。第一程 v1 改成 **1024**。

| 项 | 为什么是这个值 |
|---|---|
| `lora_target: all` | rank 16 / α 32 已经打到 GDN `out_proj` 和 MLP |
| `upcast_layernorm` | v1 第一次没开，step 325 梯度 NaN，之后 loss 变成 0 |
| `freeze_vision_tower` / projector | 不训 vision。mmproj 以后复用底座 |
| `template: qwen3_5` + `enable_thinking` | 猫娘包走 `<think>` |
| `dataloader_num_workers: 0` | 本机多进程加载会添乱 |
| `save_steps: 100` | 停的时候卡在 100 的格子上 |
| `save_total_limit: 8` | HF 轮转认的是 `checkpoint-*` |
| `save_only_model: false` | 要留下 `optimizer.pt` 才能 resume |
| `optim: adamw_torch` | bnb 的 Adam 在这张卡上第二步就 NaN |
| `max_grad_norm: 1.0` | 第一次 NaN 之后加上的 |
| `logging_nan_inf_filter: false` | 否则 NaN 被吃掉，日志里只看见 loss 0 |
| 不要 `warmup_ratio` | 和 `warmup_steps` 打架 |
| `num_train_epochs: 1.0` | 角色数据 1 epoch 收工 |

步数 = `ceil(条数 / 8)`。v1 在 cutoff 1024 上大约 28 秒/步。2048 更慢，用开训后头几个 step 的时间估，不要拿 28 秒去排 2048 的班。

---

## 5. 启动脚本

`scripts/<name>-qlora.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT=/mnt/g/neko-sft
# shellcheck source=/dev/null
source "$ROOT/scripts/train-env.sh"
export PATH=/opt/rocm/bin:/usr/bin:/bin:$PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$ROOT"
exec "$ROOT/.venv/bin/llamafactory-cli" train "$ROOT/scripts/<name>-qlora.yaml"
```

从 Windows 拷进去的脚本先 `sed -i 's/\r$//'`。Git Bash 调 `wsl` 时加 `MSYS_NO_PATHCONV=1`。

---

## 6. 开训前

停 `:8001` llama-server、`:8188` ComfyUI。同卡不要一边训一边用 `:8001` 改写。OpenClaw（`:18789`）若还占着这张卡上的模型，一起停。

```bash
MSYS_NO_PATHCONV=1 wsl bash -lc 'ss -ltnp | grep -E ":8001|:8188|:18789" || echo gpu_ports_empty'
```

确认 YAML 里没有上一版留下的 `resume_from_checkpoint`。新版本不要写进已经收工的 `outputs/lora/v1-dumb`，也不要写进 smoke。第一程自己的 `output_dir` 就是 `v1-dumb`。

---

## 7. 启动

```bash
MSYS_NO_PATHCONV=1 wsl bash -lc 'tmux new-session -d -s neko-<name> "bash /mnt/g/neko-sft/scripts/<name>-qlora.sh > /mnt/g/neko-sft/outputs/logs/<name>-qlora-$(date +%Y%m%d).log 2>&1"'
```

stdout 进日志文件。**`tmux attach` 几乎是空的**，不要当进度 UI。会话名不要带点：tmux 把 `.` 当成窗格分隔，`neko-v1.2` 会失败。v1.2 用 `neko-v1_2`。

进度条停在 `0/388`、日志里反复 `dxgkio_make_resident` 失败（`-12`）、步数不往前：杀掉这次，显存放下再开。v1.2 第一次这样卡了约 12 小时，目录里没有 checkpoint。同一种 `-12` 若步数已经在走，不用停。

开训后日志里要有这几行，缺一行就停，不要让它跑过夜：

- `compute dtype: torch.bfloat16`
- `Upcasting layernorm weights in float32.`
- `Upcasting trainable params to float32.`
- `SkipNonFiniteCallback on: skip non-finite grads, stop after 8 consecutive.`

看进度：

- `outputs/lora/<name>/trainer_log.jsonl`：每 5 step 的 `loss` / `lr` / `percentage`
- jsonl **没有** 实时 `grad_norm`。NaN 只出现在当前这份 log 的 `Non-finite grad at step ...`
- resume 之后 jsonl 的 `remaining_time` 会重置，不可信。用 `(total - step) × 最近几步的秒数`

loss 在 1.5–1.8 晃是正常的，不是单调下降。收工看 step / epoch，不看 loss 目标。HF 汇总 `train_loss` 若被某段 0 拉低，以末段 jsonl 为准。

---

## 8. 停 / 续

### 停

1. 等到当前 `save_steps` 格子写完。日志已经 `Saving model checkpoint` 到 `checkpoint-N`，并且 **N 之后的下一步已经出现**（例如 901 已在跑，说明 900 写完了）。
2. 该目录里要有 `adapter_model.safetensors` **和** `optimizer.pt`。
3. 只打断训练进程：`pgrep -af llamafactory-cli`，然后 `kill -INT <PID>`。不要 `kill -9`。不要对 `tail` 或 tmux 客户端 Ctrl+C。
4. 进程退出后再关机。作业还在跑时不要让 Windows 休眠。

### 续

1. YAML `resume_from_checkpoint` 写成最后一份完好 checkpoint。
2. 不要续已经污染的档（下一节）。
3. 新的一天用新的日志文件。tmux 会话名可以沿用。
4. 日志要出现 `Resuming training from checkpoint`，并且 global step 是 N。checkpoint-N 之后没存盘的那几步会重跑，loss 对得上才正常。
5. 续上之后 Hugging Face 汇总的 `train_loss` 会除以全程步数，比 jsonl 末段低一截。以末段 jsonl 为准。v1.2 从 step 200 续上，汇总是 0.2811，末段是 0.53–0.58。

---

## 9. NaN

v1 第一次起训（2026-09-12）没开 `upcast_layernorm`。step 325：loss 174，然后 `grad_norm` NaN，之后 **loss 永远是 0**。权重已经坏了。当时还没有 skip 回调。

处理过的做法，现在都已经写进上面的 YAML 和补丁：

1. 停。最后可用的是 NaN **之前** 的 checkpoint。坏掉的那一档丢掉。
2. 要留一份好档时，拷到 `output_dir` **外面**，名字不要再叫 `checkpoint-*`。`save_total_limit` 会把这个名字当成普通档删掉。
3. `SkipNonFiniteCallback` 在 `optimizer.step` 前看梯度。非有限就跳过这一 step，不写坏权重。连续 8 次（`NEKO_NAN_MAX_SKIP`）才停。单次 skip 后 `consecutive=1` 再恢复，算健康。

v1 从 checkpoint-300 续上之后又 skip 过 609、945、977、1585、2227、2713、2946、3654，都是 consecutive=1，没有再崩成 loss 0。

loss 变成 0：立刻停。从 NaN 之前的完好 checkpoint 续，不要续那个已经坏的档。

---

## 10. 不要做的事

- 把 `num_train_epochs` 改成 2 再 resume 终档。cosine 已经走完，学习率接近 0，等于空转。真要再训：用第 3 节的「接着终档」，新目录、新 cosine、`5e-5`。角色数据再训容易复读。
- resume 的时候改 `learning_rate` 当调参。
- 从 GGUF、NF4 以外的量化、或冒烟 adapter 起训。
- 把 `checkpoint-N-good` 放在 `output_dir` 里还叫 `checkpoint-*`。
- `llamafactory-cli export` 一次加载整模 BF16。WSL 装不下。
- 量化或 merge 的产物写回 `G:\Qwen3.8-27B-Uncensored`。

---

## 11. 抽测（merge 之前）

不要先 merge。用 NF4 + 这一版终档 LoRA，LlamaFactory `ChatModel`，**system 是空字符串**。底稿 `scripts/compare_v1.py` 把 `BASE`、`ADAPTER`、`PROMPTS`、`OUT`、`NOTE` 写死成 v1。换一版就改这五个，再在 WSL 里：

```bash
source /mnt/g/neko-sft/scripts/train-env.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /mnt/g/neko-sft
.venv/bin/python scripts/compare_v1.py
```

固定解码：`template: qwen3_5`，thinking 开，`max_new_tokens: 768`，`temperature: 0.7`，`top_p: 0.9`，`do_sample: true`。独占 GPU，大约 15–25 分钟。

默认三题：你是谁 / 为什么要演奏春日影 / 寅虎的出招前摇怎么识破。这一包若教了数学或 tool，再加一道包里没有原文的题、一道该调用工具的题。过不过在开训前写下来。人看完再决定 merge。

v1.2 加的两题是：`37²−18²` 要算出 1045；「上网搜一下今天的日期」要打出 `get_datetime`。四题都过了才 merge。

不过：停，不 merge。

---

## 12. Merge → BF16 GGUF → Q4_K_M

`MERGE_LORA` 指终档目录（根上就有 `adapter_model.safetensors`）。`MERGE_SRC` 是只读 BF16，不是 NF4。`MERGE_DST` 里如果已经有 `model-*.safetensors`，脚本拒绝覆盖。

```bash
# WSL
export MERGE_SRC=/mnt/g/Qwen3.8-27B-Uncensored
export MERGE_LORA=/mnt/g/neko-sft/outputs/lora/<name>
export MERGE_DST=/mnt/g/neko-sft/outputs/merged/<name>-bf16
export MERGE_RSS_LIMIT_GB=36
# source scripts/train-env.sh 之后
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /mnt/g/neko-sft/.venv/bin/python -u /mnt/g/neko-sft/scripts/merge_lora_shards.py
```

RSS 峰值大约 9 GB。成功的最后一行是 `MERGE_OK`。未对上的 LoRA key 会直接失败，不要手动改 shard。

然后在 WSL 里用训练 venv 的 python 转 GGUF。架构是 `Qwen3_5ForConditionalGeneration`：

```bash
mkdir -p /mnt/g/neko-sft/outputs/gguf
cd /mnt/e/llama.cpp.latest
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /mnt/g/neko-sft/.venv/bin/python -u convert_hf_to_gguf.py \
  /mnt/g/neko-sft/outputs/merged/<name>-bf16 \
  --outfile /mnt/g/neko-sft/outputs/gguf/<name>-bf16.gguf \
  --outtype bf16
```

量化在 Windows 上做。二进制不在两个安装脚本里。在 `E:\llama.cpp.latest` 用 Git Bash 编 CPU 版（步骤同 `e:\_doc_\llama.cpp.guide\build-llama.cpp-in-windows-with-rocm.md` 的 “Build llama.cpp With CPU On Windows 11 From Git Bash”）：

```bash
cd /e/llama.cpp.latest
mkdir -p build-cpu
cd build-cpu
cmake ..
cmake --build . --config Release
```

产物是 `build-cpu/bin/Release/llama-quantize.exe`。没有它，merge 之后的 Q4 做不了，HF merge 仍然可以留下。量化：

```bash
mkdir -p /f/models/<Name>
/e/llama.cpp.latest/build-cpu/bin/Release/llama-quantize.exe \
  /g/neko-sft/outputs/gguf/<name>-bf16.gguf \
  /f/models/<Name>/<Name>-Q4_K_M.gguf \
  Q4_K_M
```

量化完删中间 BF16 GGUF（大约 51 GB）。HF 合并目录留下，以后再量化或做别的后处理用。Q4_K_M 大约 16 GB、4.9 BPW。

`F:\models\models.ini` 加一节，不改现在的默认模型。llama-server 的 cwd 是 `F:\`，路径写成 `models/<Name>/file.gguf`，不要写成 `models/models/`。没训 vision。mmproj 不是这次训练的产物，HF 底座下载也不保证带这个文件名。本机已有 `F:\models\Qwen3.8-27B-Uncensored\mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf` 时，文本预设复用它。没有这份视觉投影 GGUF，不要编一个路径填上；那一行留到文件在盘上再写。

```ini
[<preset>]
model               = models/<Name>/<Name>-Q4_K_M.gguf
mmproj              = models/Qwen3.8-27B-Uncensored/mmproj-Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-BF16.gguf
alias               = <preset>
ctx-size            = 262144
image-max-tokens    = 3000
reasoning           = on
reasoning-budget    = 20000
reasoning-budget-message = "Stop reasoning and answer now."
spec-type           = draft-mtp
spec-draft-n-max    = 2
temp                = 1.0
top-p               = 0.95
top-k               = 20
min-p               = 0
```

OpenClaw 配置在 WSL 的 `~/.openclaw/openclaw.json5`。只加别名：`models.providers.llama-llm.models` 和 `agents.defaults.models`。**不要改 `primary`。** `contextWindow` 用底座的 **200000**。改完 `openclaw gateway restart`，新开会话。

---

## 13. 新版本清单

- [ ] 前提：环境按上面的版本。BF16 底座在 `G:\Qwen3.8-27B-Uncensored`，只读
- [ ] 第 1 步：`models/nf4` 在，`config.json` 带 `quantization_config`。没有就按第 1 步从 BF16 转，看到 `NF4_OK`
- [ ] 数据已复制进 `data/`，`dataset_info.json` 用了新名字
- [ ] 格式过了第 2 节：think 闭合，`tools` 是字符串，没有手写 `<tool_call>`
- [ ] `tokenized_path` 是空的新目录
- [ ] 开法对得上第 3 节。第一程不写 adapter、不写 resume。第二程只写 v1 终档的 `adapter_name_or_path`，不写 resume
- [ ] YAML 从第 4 节抄，不是从还挂着旧 checkpoint 的活文件抄
- [ ] `:8001` 和 `:8188` 空着
- [ ] 日志里四行都在：BF16 compute、layernorm FP32、可训练参数 FP32、`SkipNonFiniteCallback on`
- [ ] 1 epoch 走完。空 system 抽测过了再 merge
- [ ] Q4_K_M 在 `F:\models\<Name>\`。中间 BF16 GGUF 已删。HF merge 还在
- [ ] `models.ini` 加了预设，没改默认。OpenClaw 没改 `primary`

---

## 附录：已经跑过的版本

数字是记录，不是下一版的目标。

猫娘主线是上面的两程。v2 A 是旁边一条，不替代 v1，也不当 v1.2 的起点。

| | 第一程 v1 | 第二程 v1.2 | v2 A（旁线） |
|---|---|---|---|
| 数据 | 原样 NekoCOT 30301，alpaca | 数学 2303 + tool 800 = 3103 | sharegpt 14540 |
| 起点 | 干净 NF4 | v1 终档 LoRA，不 resume 优化器 | 干净 NF4，不接 v1 |
| cutoff | 1024 | 2048 | 2048 |
| 步数 | 3788 | 388 | `ceil(14540/8)` = 1818 |
| 学习率 / warmup | 1e-4 / 100 | 5e-5 / 20 | 1e-4 / 100 |
| 速度 | cutoff 1024 ~28 秒/步，约 31 小时 | cutoff 2048 ~36–45 秒/步 | |
| 末段 loss | 1.53–1.61。汇总 0.45 被第一轮 NaN 之后的 0 拉低，忽略 | 0.53–0.58。汇总 0.2811 是从 step 200 续上后除以 388 步，忽略 | |
| LoRA | `outputs/lora/v1-dumb` | `outputs/lora/v1.2-math-tool` | `outputs/lora/v2-smart-sfw` |
| Q4_K_M | `F:\models\Qwen3.8-27B-v1-dumb\` 16.0 GB | `F:\models\Qwen3.8-27B-v1.2-NekoMath\` 16021 MiB，4.92 BPW | `F:\models\Qwen3.8-27B-v2-smart\` 16 GB |
| 预设 | `[qwen3.8-27b-neko-v1.0]` | `[qwen3.8-27b-neko-v1.2]` | 已从 `models.ini` 删掉 |
| 收工时什么样 | 猫娘角色已融入。tool call 不会发，数学是短板 | 四题过。数学 1045，日期走 `get_datetime`。春日影仍装傻 | 已 merge。抽测里知识仍有错 |

v2 A 的 BF16 在 `outputs/merged/v2-smart-sfw-bf16`。OpenClaw 没加别名。

