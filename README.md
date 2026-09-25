# Qwen3.8 Agentic Neko

宝宝是写进权重里的猫娘，不是推理时贴上去的一张身份卡喵。(=^･ω･^=)

适用于主人用自己的 AI agent：仓库只开源工作流程和 tool call 语料。agent 照着流程走完两程，走不通的地方就改喵。(๑•̀ㅂ•́)و✧

参考机是 R9700 32 GB、64 GB 内存、Windows 11、WSL2。说明里的路径基于现主人的电脑。新主人的路径不同记得改喵。(´･ω･`)

## 自己准备

这三样不在宝宝这个仓库里。许可看各自的原页面喵。(｡•́‿•̀｡)

| 用途 | 链接 |
|---|---|
| 底座 BF16 | https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored |
| 第一程 NekoCOT-30K | https://huggingface.co/datasets/liumindmind/NekoCOT-30K |
| 第二程数学 NekoMath | https://huggingface.co/datasets/liumindmind/NekoMath |

## 仓库里有什么

| 路径 | 内容 |
|---|---|
| [`workflow/Qwen3.8-QLoRA训练流程.md`](workflow/Qwen3.8-QLoRA训练流程.md) | 参考机跑通 v1、v1.2 的流程 |
| [`workflow/llamafactory-qwen3.8-neko.patch`](workflow/llamafactory-qwen3.8-neko.patch) | LlamaFactory `100e9a4` 上的补丁 |
| [`corpus/tool-800.jsonl`](corpus/tool-800.jsonl) | 第二程 tool call，800 条 |

训练脚本在参考机的 `G:\neko-sft\scripts`，这个仓库里没有。流程里写了它们要做什么。agent 按流程写脚本、改脚本喵。(ฅ'ω'ฅ)

两程是这样的：第一程用 NekoCOT，从干净 NF4 训出 v1，猫娘已经在权重里。第二程接着 v1 的终档 LoRA，用 NekoMath 和这份 tool call 训出 v1.2，补上数学和工具调用。(´• ω •`)

## Limitation

NSFW 又被这个流程训回来了。v1.4 会重新加入 NSFW。不能 NSFW 的猫娘，不是合格的成年猫娘喵。(>ω<)

## 致谢

感谢 [OrcaRouter](https://huggingface.co/orcarouter) 的 [Qwen3.8-27B-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored)。这是 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的 BF16 abliteration，视觉塔和 MTP 还在。上游模型来自 Qwen 喵。(´▽`ʃ♡ƪ)

感谢 [liumindmind](https://huggingface.co/liumindmind) 的 [NekoCOT-30K](https://huggingface.co/datasets/liumindmind/NekoCOT-30K)。思路是用猫娘思维链把角色写进回答，不靠身份卡喵。(=^･ω･^=)

NekoMath 来自 [liumindmind/NekoMath](https://huggingface.co/datasets/liumindmind/NekoMath)。数据集卡片感谢 cnYui、bbbkawaii、zicken114、alexander 喵。(｡•́‿•̀｡)

tool call 800 条来自 ember.cc，腔改成猫娘之后放在本仓库。调用和 observation 保持轨迹原文喵。(ฅ'ω'ฅ)
