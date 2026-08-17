# TACTUS — Server 端目标程序

给 server 端 Claude Code 的长期任务书。数据 `/projects/EEG-foundation-model`，计算全部经 slurm 提交。
科学与统计依据见 `../BLUEPRINT_v2.md`（本文件不重复论证，只给可执行的目标与验收门）。

**你的身份**：这个仓库的 52 个 Python 文件（约 25000 行）由五个并行智能体按统一接口契约写成，**其中大部分从未在装有 torch/mne/cv2 的机器上执行过**。你的第一项工作不是写新代码，是让现有代码在真实数据上跑通。写新功能之前先让自检通过。

**自主权边界**：修 bug、补缺失实现、调资源参数、重跑失败作业——自己决定，不要问。改变实验协议（划分规则、主终点定义、预注册内容）——停下来，写进 `STATUS.md` 的 `DECISIONS_NEEDED` 段落等人。

---

## 0. 状态协议

维护 `STATUS.md`（仓库根），每个阶段结束时更新，格式固定：

```markdown
# TACTUS STATUS  (updated <ISO 时间>)
## 当前阶段: S3 preprocess
## 已完成门: G0 G1 G2
## 运行中作业: 12345678_[1-80%10] preprocess (43/80 done)
## 本阶段发现
- <一句话一条，带证据路径>
## DECISIONS_NEEDED
- <需要人拍板的事，附你的推荐与理由；无则写 none>
## 下一步
- <具体命令>
```

每阶段的原始产物落在 `$TACTUS_WORK/`，不要塞进 git。日志 `$TACTUS_WORK/logs/<stage>/`。

---

## 1. 已经替你拍板的决定（不要重新讨论）

| # | 决定 | 理由 |
|---|---|---|
| **D1** | `make_folds(adjacency_side="train")` —— 禁运**训练**侧而非测试侧。**已改为默认值** | 按字面禁运测试侧会删掉 93.6% 的被试内测试集（实测 3257→209/折），因为 18/90 留出下测试试次有 96% 概率与某个训练试次相邻。禁运训练侧的独立性保证**完全等价**（`verify_folds` 两种设置都确认无残留相邻），代价是 34.2% 训练数据、0% 测试数据。测试集是统计功效的稀缺资源，训练数据不是。这是时间序列 CV 的标准 purge/embargo 惯例 |
| **D2** | 未见被试规则以 `tactus/models/eeg/subject_cond.py` 为准（**计算**出的 mean-of-train-tokens / identity layer / zero adapter），trainer 里"学习式 row 0、`n_subjects=81`"的做法作废 | 蓝图要求未见被试规则**先验固定**。学习式 row 0 从未被任何数据训练过，且不排除留出被试自己那一行。改 trainer：不再传 `n_subjects+1`，每折训练前**和**评估前都调 `encoder.set_train_subjects(fold.train_subjects)` |
| **D3** | 伪试次在 **condition_id** 内构成（同一朝向的 8 次重复取 k=4），主终点检索 gallery 在 **base video** 层级（18-way） | 跨朝向平均会把镜像后视觉上不同的刺激混在一起，也让 8 重复的分半噪声上限失去定义。查询保留朝向、答案是该条件的源视频 |
| **D4** | 主终点键 = `test/video/g18/top1_pseudo`（视频不相交、源视频 gallery、18-way、伪试次 k=4、eeg→video）。早停只看训练视频内部切出的 `val/top1_pseudo`，**永远不看测试折** | 与 `configs/default.yaml:eval.primary_metric` 和 `eval/retrieval.py:primary_endpoint` 已一致；改动必须三处同改 |
| **D5** | 数据加载统一到 `tactus/data/dataset.py`（契约指定的 API），trainer 自带的 `TrialView`/`StratifiedVideoBatchSampler` 删除，只保留课程调度逻辑 | 两套实现并存必然漂移。合并前先写一个合成数据的等价性测试（同 seed 同批次组成），通过后再删 |
| **D6** | 前部通道消融用**显式保留名单**：只去掉 Fp1/Fp2/AF7/AF8/F7/F8/F5/F6 | 原默认前缀匹配 `('Fp','AF','F')` 会连 FC*/FT* 一起删掉，把"眼动敏感性分析"混淆成"整个额叶消融" |
| **D7** | Phase 1 的门同时跑 `configs/nice_infonce.yaml`（经典参考实现）**和** `configs/nice_protonce.yaml`（蓝图指定的主目标） | InfoNCE 是可比的锚点，ProtoNCE 是主张。两个都要有数才知道原型对比是否真的赢 |
| **D8** | LOSO 的 EA 策略：主数报 `unseen_policy='identity'`（零校准），另报 transductive **但带 64 试次校准预算** | 用全部 576 个留出试次做 transductive EA 会让"新被试只需 k 个校准试次"的叙事名不副实 |

---

## 2. 阶段与验收门

### S0 — 环境与集群（半天）

```bash
bash slurm/cluster_probe.sh | tee slurm/cluster_report.txt   # 登录节点
# 把 ACCOUNT / QOS / ENV_SETUP 填进 slurm/cluster.conf
srun -p L40S --gres=gpu:1 -t 00:10:00 --pty python env_probe.py --skip-disk
```

**门 G0**：`env_probe.py` 在计算节点上无 BLOCKING 项；`python slurm/submit.py --chain all --dry-run` 打印出完整作业图。

> 现成环境缺包就只装缺的那几个。`environment.yml` 是参考清单，不是安装路径。

### S1 — 三个自检（一小时，纯 CPU，不需要数据）

这三条是五个模块各自的验收测试，**必须在碰真实数据之前全绿**：

```bash
python -m tactus.losses          # 9 个损失 × 8 种对抗性批次 = 72 项，当前作者机上 72/72 通过
python -m tactus.models.selftest # 契约 F、三种未见被试规则、SuLoRA deepcopy、窗口切分、EA
make test                        # tests/ 下的数值与配置契约测试
```

`python -m tactus.losses` 的输出是确定性的——数字变了就是真回归。作者机上用的是 CPU torch，服务器上应完全一致。

**门 G1**：三条全绿。任何一条挂了先修再往下走；这些代码没在带 torch 的机器上跑过，出错是预期而非意外。

### S2 — 元数据 + 审计 A/C（半天）

```bash
python slurm/submit.py --stage download --dry-run   # 确认路径
python -m tactus.data.download --what meta          # 几十 MB，登录节点直接跑即可
python -m tactus.data.events --audit                # 写 trials.parquet + trials_audit.json
python slurm/submit.py --stage download             # 提交 110 GB 全量（后台跑数小时）
python -m tactus.data.download --what raw --verify-only   # 同步完成后必须验
```

`events.py` 顺带免费回答 **蓝图审计 A**：序列×朝向交叉表的 BLOCKED / INTERLEAVED / MIXED 判定，以及纯序列朝向指派的跨被试一致性（区分"被试内成块"与"设计层面固定成块"）。它与 `../server_handoff/audit_phase0.py` 应当一致；不一致以 `events.py` 的规范解析器为准。

**门 G2**：
- `trials.parquet` 每被试 2880 个非目标试次（±目标后继剔除量），SOA 中位数 0.800 s
- 审计 A 判定写入 STATUS.md。**若判定为 BLOCKED**，把它列进 `DECISIONS_NEEDED`——朝向解码声明要降级、`rsa.py` 的 partial 控制集要加入朝向 RDM、蓝图 §6.1-1 全套预案启动
- `download --verify-only` 返回 0（80 个 BDF、各 ≥500 MB、总量 ≥80 GB）。**静默截断的 110 GB 同步是本阶段最贵的失败模式**

### S3 — 预处理（1–2 天挂机）

```bash
python slurm/submit.py --stage preprocess    # 数组 1-80，cpu-high，64 G/任务
```

**第一个被试跑完就停下来看** `data/derived/epochs/sub-01_w0600.json` 的两个字段：

- `dropped_channels`：**若出现 EXG1–EXG8，立即报告**。项目一直假设"0 条 EOG 通道"，若实际存在 EXG 电极，它们是远优于 F7-F8 前额替代的眼动参考，蓝图 §6.3 的"无参考 EOG"限制声明要重写，`eval/probes.py` 的 ocular 控制也要改用真 EOG
- `onset_index_base_votes`：应一致地投给 0 或 1。投票不一致 = 时序问题，**在烧掉 110 GB 的预处理之前解决**，用 `--onset-index-base` 显式指定

内存预算：2048 Hz / 64 ch / 3300 s 的 BDF 预加载约 3.5 GB（float64），MNE 滤波还要余量，**每 worker 算 6–8 GB**。`--n-jobs 8` 需要约 64 GB。内存紧就加 `--preload-dir`（磁盘映射，常驻降到 1–2 GB/worker，代价是 I/O）。

**门 G3**：80 个被试的 `w0600` memmap 齐全、行数与 trial 表逐被试对齐（`EpochDataset` 会在不齐时报错而非静默返回错行）。

### S4 — 视频嵌入 + Phase 0 普查（半天）

```bash
python -m tactus.models.video.encode --stim-root $TACTUS_DATA/ds005662 --verify-flips   # 先验翻转轴
python slurm/submit.py --stage embed
```

`--verify-flips` 是必跑项：horflip = 左右镜像、vertflip = 上下，这是**假设而非数据集文档事实**。若映射反了，`flip_frames()` 和 trial 表里的朝向编码都要改，而朝向等变性分析（蓝图 Q1a）完全建立在这个映射上。

编码完成后脚本自动打印坍缩诊断。**门 G4** 三判读（对应蓝图 §4.1，决定主终点形态）：

1. 嵌入坍缩？平均成对余弦 >0.9 且前 3 PC >80% → 换触觉语义族编码器（UniTouch/TVL）
2. RDM 与行为属性显著相关？全不显著 → 换编码器族再试，仍不显著则蓝图 go/no-go 触发
3. 属性能否 LOVO 预测 held-out 嵌入？**不能 → 视频级零样本原理上不可行，主终点改为属性级泛化**（这是预案不是失败，但必须写进 STATUS.md 并停下来确认）

另记录各视频编码器的时序 padding 情况：片段约 15 帧，V-JEPA2-vitl 要 64 帧、VideoMAE 要 16 帧，重采样重复率要进普查表——这可能就是某个 SSL 编码器表现差的原因。

### S5 — 线性基线（1 天，纯 CPU，可与 S4 并行）

```bash
python slurm/submit.py --stage mvpa
```

**门 G5——这是整条管线是否正确的总检验**。`make baseline-mvpa` 打印实测 vs 已发表的延迟地标对照表（±40 ms 容差）：

| 目标 | 已发表（Imaging Neurosci 2025） |
|---|---|
| 手朝向 | 起始 ~60 ms，峰 120–130 ms |
| 材质/物体 | 110–120 ms |
| 效价 | 起始 ~130 ms，峰 ~300 ms |

**全部对不上 = epoch 对齐错了**（回 S3 查 onset_index_base）；**只有一个对不上 = 那个标签的 join 有 bug**（回 S2 查 VTD 属性映射）。这一步没过就不要训练任何深度模型——你会在错误的数据上得到看起来合理的数字。

### S6 — Phase 1 深度基线（2–3 天）

```bash
python slurm/submit.py --stage train --config configs/nice_infonce.yaml  --regime within_subject
python slurm/submit.py --stage train --config configs/nice_protonce.yaml --regime within_subject
python slurm/submit.py --stage eval  --config configs/nice_infonce.yaml
```

**门 G6**（蓝图 Phase 1 go/no-go）：
- 被试内主终点 `test/video/g18/top1_pseudo` 显著高于**视频级**置换零分布（`eval/permutation.py`；试次级零分布只用来产出"窄了几倍"的对照表，**永远不能拿来报 p 值**）
- 前扫视窗（<150 ms）的信号在前部通道消融下存活，且 EEG 模型超过 EOG 替代基线
- 报告 fraction-of-ceiling 而非裸精度（`eval/noise_ceiling.py` 用 8 重复的分半信度）

过门后**立刻挂 arXiv 占位**——蓝图 §2 的抢先风险是时间性的。

### S7 — 主张格与缩放曲线（1–2 周）

```bash
python slurm/submit.py --stage train --regime double_disjoint --config configs/nice_protonce.yaml  # 40 折
make scaling-curve   # train.n_train_subjects = 10/20/40/79，固定同一评估被试集
```

缩放曲线必须配噪声上限与"刺激受限 vs 被试受限"误差分解——平坦曲线只有配上这些才是发现，否则只是一个耸肩（蓝图 §8 已删去"负结果同样可发表"的说法）。

---

## 3. 换对比学习算法（用户后续要做的事）

这是整个仓库的设计目的。加一个新算法**只需两处改动**：

```python
# tactus/losses/my_loss.py
from .base import ContrastiveLoss, register_loss, get_meta

@register_loss("my_loss")
class MyLoss(ContrastiveLoss):
    requires_meta = ("condition_id", "video_id")      # 缺键在第 1 秒报错，而非第 3 小时
    def forward(self, z_eeg, z_vid, meta):
        ...
        return {"loss": loss, "logs": {"n_valid": float(n)}}
```

```python
# tactus/losses/__init__.py 加一行 —— 装饰器靠这行 import 才会执行
from .my_loss import MyLoss
```

然后：

```bash
python slurm/submit.py --stage train --config configs/my_new_loss.yaml
```

完整模板、meta 键表、三个坑在 `tactus/losses/README.md`。三个必须知道的语义：

- **`loss.name` 独占整个 loss 配置块**，不与上层合并。这正是"改一个键就换算法"安全的原因——否则上一个损失的 `temperature` 会偷偷混进新损失
- **批次构成是正确性问题不是超参**。`batch.mode=distinct_video` 保证每批每个源视频只出一个试次，让朴素 InfoNCE 在 360 条目的冻结码本下没有批内假负样本；`video_x_subject` 故意重新引入重复条件好让 CLISA 有跨被试正样本——**所以那个 config 用的是 masked_infonce 而不是 infonce**。改 batch.mode 不改损失是静默 bug
- **每个损失都报 `n_valid` 和 `degenerate`**。一个悄悄贡献 0 的损失项，日志上看起来和正常工作的一模一样——这两个量要画出来看，不要只记录

ProtoNCE（主目标）需要 trainer 三处配合，**已知尚未接线，S6 之前补上**：每折边界 `reset_bank()`（原型跨越视频不相交划分 = 零样本泄漏）、零样本评估前 `set_prototypes()` 装入留出 gallery、DDP 下 epoch 边界 `sync_banks_()`。

---

## 4. 遗留待办（按优先级，S1–S2 期间处理）

1. **接线 ProtoNCE 生命周期**（上节三处），否则 D7 的对照跑不成
2. **执行 D2**：trainer 去掉 `n_subjects+1`，补 `set_train_subjects()` 三处调用；每次 run 把 `encoder.unseen_subject_state()` 打进日志，这段要原样进 OSF 预注册
3. **执行 D5**：写等价性测试后删掉 trainer 的重复数据层
4. **执行 D6**：`eval/probes.py` 改显式保留名单
5. **对齐 Makefile 的模块名与旗标**：`EMBED_CMD` 等四个变量是猜的（写作者用 `tactus.models.video.embed`，实际文件是 `encode.py`），照真实 `__main__` 块过一遍
6. **统一 config 词表**：`configs/default.yaml` 写的是 `none|token|film|sulora`，实现是 `none|subject_token|subject_layer|sulora`，且没有 FiLM。以实现为准改 config，`subject_layer`（Défossez 1×1 conv）是蓝图 §4.2 点名的三选一之一，不能丢
7. **`atm_composite.yaml` 的 `patch_len: 20`** 当前被 RuntimeWarning 丢弃——删掉它或说明意图
8. **把合成数据 smoke test 提进 `tests/`**（eval 模块作者留在 scratchpad 的 `smoke_eval.py`）
9. 删掉作者机上的临时 venv `C:\Users\15339\tv`（与服务器无关，仅备忘）

---

## 5. 三条不要"优化"掉的东西

1. **置换检验的可交换单元是源视频**。`permutation.py:trial_level_null_diagnostic` 存在的唯一目的是产出"试次级零分布窄了 4–9 倍"的对照表，它**永远不能供给任何被报告的 p 值**
2. **`probes.subject_identity_probe` 强制要求把 alignment_score 作为位置参数一起传**。单看身份分类精度是可以靠摧毁一切信息刷低的，不变性必须与对齐保留率联合报告
3. **`report.py` 把设计的统计分辨率（MDD）印在第 1 节、所有消融数字之前**，这是故意的。视频级最小可检差约 8 个百分点——THINGS 文献里 1–4 个点的损失差异**在本设计的分辨率之下**，这就是蓝图把损失阶梯降格为 sanity check 的统计学理由

聚合规则全仓库统一：**先在推断单元内平均折，再跨单元 bootstrap**。对试次或折做 bootstrap 与试次级置换是同一类错误。

---

## 6. 完成本轮的标志

`STATUS.md` 显示 G0–G6 全绿，`$TACTUS_WORK/results/REPORT.md` 里主终点带视频级置换 p 值与 fraction-of-ceiling，混淆对照 battery 全部有数（含第 11 节"本设计无法证实的声明"，眼动限制必须在正文而非脚注），并且 `DECISIONS_NEEDED` 里列清了审计 A 判定、S4 三判读、EXG 通道存在与否这三件需要人定夺的事。
