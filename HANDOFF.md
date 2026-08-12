# 交接说明 — 8 卡预训练

这是一个 150M 语言模型的预训练实验：三个模型，除了训练数据里推理数据的占比
（0% / 15% / 30%）之外**其他一切完全相同**，用来看这个占比如何影响模型的输出行为。

已经在单张 4090 上跑通过小规模版本，这次是放大约 9 倍。所有代码都调试过，
你不需要理解实验内容 —— 按顺序跑五条命令，把指定的东西发回来就行。

---

## 需要的资源

| | |
|---|---|
| GPU | 8 张（H800/A100/H100 均可），**实际占用约 8–10 小时** |
| 磁盘 | **约 250 GB**，在 `workspace/` 下 |
| 时间 | 总跨度约 8–12 小时，其中数据准备约 1 小时是**纯 CPU**，那段时间 GPU 空着 |
| 外网 | 需要（下载数据集和依赖） |

---

## 五步

每一步跑完把 **「发回」** 那一列的东西给我就行，不用判断对错。

| 步骤 | 命令 | 时间 | 需要 GPU | 发回 |
|---|---|---|---|---|
| 0 | `bash run.sh probe` | 30 秒 | 否 | 整个输出 |
| 1 | `bash run.sh env` | ~25 分 | 否 | 最后 30 行 |
| 2 | `bash run.sh smoke` | ~15 分 | **是** | 整个输出 |
| 3 | `bash run.sh data` | 30–60 分 | 否 | `workspace/status.txt` |
| 4 | `bash run.sh train all` | 6–9 时 | **是** | `workspace/status.txt` |
| 5 | `bash run.sh eval` | ~30 分 | **是** | `workspace/status.txt` |

### 开始之前

```bash
git clone -b echo-chamber-repro https://github.com/DanielHuangjiakang/openrlhf-pretrain.git
cd openrlhf-pretrain
```

**不需要 HuggingFace 账号、token 或任何登录。** 全部资源都是公开可匿名下载的，
已经逐个验证过。

### 数据从哪来

不需要你手动下载任何东西。`bash run.sh data` 会自动从 HuggingFace 拉取并直接分词
（datatrove 流式读取，边下边处理）：

| 数据集 | HuggingFace 仓库 | 取多少 |
|---|---|---|
| TinyGSM | `TinyGSM/TinyGSM` | 全部 17 个分片 |
| FineMath-3+ | `HuggingFaceTB/finemath` | 16 / 128 个分片 |
| Algebraic-Stack | `EleutherAI/proof-pile-2` | 32 / 79 个分片 |
| tokenizer | `NousResearch/Llama-2-7b-hf` | 仅词表文件 |

只取需要的分片，不是整个数据集 —— 所以是 ~80 GB 而不是几个 TB。

**下载缓存会放在 `workspace/hf-cache/`**，不是默认的 `~/.cache/huggingface`。
这是故意的：集群上家目录通常有配额，80 GB 会把它撑爆。所以确认 `workspace/`
所在的分区有 250 GB 就够了，不用管家目录。

如果你的机器需要走代理才能访问 HuggingFace，设好 `HTTPS_PROXY` 再跑；或者
设 `export HF_ENDPOINT=https://hf-mirror.com` 走国内镜像。

### 第 2 步是关键 —— 请务必跑

`smoke` 会用极小的数据把**整条链路**走一遍：装好的环境、数据处理、
多卡通信、50 步真实训练。它失败的代价是 15 分钟，跳过它直接跑第 4 步、
失败的代价是好几个小时。

它跑完会打印实测吞吐（`tokens_per_second`），我用那个数字预估第 4 步要多久。

### 第 3 步不占 GPU

下载和分词是 CPU 密集的。这一步跑的时候 GPU 完全空闲，你可以拿去做别的事。
中断了直接重跑同一条命令，会接着来，不会从头下。

### 第 4 步可以中断

训练每 1000 步存一次盘。如果任务被杀、机器重启，**重跑 `bash run.sh train all`
即可**，它会自动从最新的存档接上，不会从零开始。

---

## 随时查看进度

```bash
bash run.sh status
```

会打印 GPU 占用、每个模型跑到第几步、磁盘剩余。

想实时看某个模型：

```bash
tail -f workspace/logs/train-tg30-8b.log
```

---

## 出问题了

所有阶段失败时都会打印一行 `!! ...`，告诉你该发哪个日志文件。把那个文件
的**最后 100 行**发回来即可：

```bash
tail -100 workspace/logs/<文件名>
```

几种常见情况：

| 现象 | 处理 |
|---|---|
| 第 1 步装依赖失败 | 把完整控制台输出发回来，通常是 CUDA/torch 版本要调 |
| 第 2 步多卡训练卡住不动 | 大概率是 NCCL。发回 `workspace/logs/smoke-train.log` |
| 显存不足 (OOM) | 把三个 `pretraining/configs/RL-150M-tg*-8b.yaml` 里的 `device_train_microbatch_size: 32` 改小成 16 或 8。**这不会改变实验结果**，只影响速度 |
| 磁盘满 | 发回 `df -h`，我来缩减数据量 |

---

## 唯一一件请不要改的事

三个配置文件里都有 `global_train_batch_size: 256`。

**这个数字不能改。** 它决定了优化过程本身，改了之后新旧实验就没法比较，
整个实验的意义就没了。想让 GPU 跑得更满，请改
`device_train_microbatch_size`（上面 OOM 那一行说的就是它）—— 那个参数
在数学上完全等价，只影响显存和速度。

配置文件顶部也写了这条提醒。

---

## 这在跑什么（可选阅读）

三组数据的唯一区别是 TinyGSM（一个"用 Python 函数解小学数学题"的合成数据集）
占多少：

| 组 | TinyGSM | 背景语料 | 总量 |
|---|---|---|---|
| `tg00` | 0% | 100% | 8,860,467,200 tokens |
| `tg15` | 15% | 85% | 8,860,467,200 tokens |
| `tg30` | 30% | 70% | 8,860,467,200 tokens |

三组的 token 总量**精确相同**（不是近似相同），所以任何差异只能来自配比。
背景语料在三组之间是嵌套的 —— 共享的部分逐字节相同。这是这个实验相对
原论文的主要改进：原论文里配比和训练量是绑在一起变的，没法分离。

先跑 `tg30`（推理数据最多的一组），因为它最可能出效果。
