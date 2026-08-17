# TACTUS STATUS  (updated 2026-08-16T15:20Z)

## 当前阶段: 基线阶梯补完 + 40 折主张格（运行中）
## 已完成门: **G0 G1 G2 G3 G4 G5**，G6 判据 1 通过 / 判据 2 修复后重跑中

## 运行中作业（都不阻塞换算法）
- `pool:autoreject` 24/80 — 独立阶段，只标注不丢弃
- `pool:mvpa_bal` — 平衡精度版 MVPA 的 video-CV 分支

---

# 一、Phase-1 主结果

`within_subject`、5 视频折、80 被试、主终点 `test/video/g18/top1_pseudo`（决定 D4）：

| 目标 | 主终点 | 95% CI | 视频级置换 p | z | fraction-of-ceiling |
|---|---|---|---|---|---|
| `nice_infonce`（参考实现） | 10.99% | 10.51–11.47 | **0.0002** | 11.5 | 0.614 |
| `nice_protonce`（蓝图主目标） | **11.93%** | 11.32–12.55 | **0.0002** | 12.6 | **0.674** |

chance = 5.56%。置换 5000 次，可交换单元 = **源视频**（p 已到 1/5001 下限）。

**试次级零分布对照（永不用于报 p 值，只用于产出这张表）**：
视频级零分布 SD 0.00433 vs 试次级 0.00105 → **窄 4.12 倍**，正落在蓝图预言的 4–9 倍区间。

**D7 两格对照（预注册的固定刺激推断：同折、被试级配对 Wilcoxon，n=80）**：
ProtoNCE − InfoNCE = **+0.94 个百分点**，bootstrap 95% CI [+0.67, +1.22]，
W=498，**p = 7.4e-8**，80 人中 59 人偏向 ProtoNCE。
**但推断目标必须写清楚**：这是**固定刺激**声明——"在这 90 个视频上 ProtoNCE 更好"。
它**不**支持"原型对比对触摸视频对齐更好"这种刺激泛化声明；后者的视频级 MDD ≈ 8 个百分点，
0.94 点远在其下（蓝图 §4.3 把损失阶梯降格为 sanity check 的统计学理由，此处实测确认）。

检索梯度（ProtoNCE 同形）：g2 → g10 → g18 单调，伪试次 k=4 一致优于单试次
（18-way：0.110 vs 0.085；72-way：0.051 vs 0.032）。

### 属性捷径量化（蓝图 §5.2 要求的那一项，新增）
只在 **material 内**置换 gallery 的 null——即"一个除了 8 类材质什么都没学到的模型"恰好落在 null 上：

| 目标 | observed | 普通 null | **material-matched null** | p | **超出材质的比例** |
|---|---|---|---|---|---|
| nice_infonce | 0.1030 | 0.0556 | 0.0676 | 0.0002 | **74.8%** |
| nice_protonce | 0.1007 | 0.0555 | 0.0682 | 0.0002 | **72.0%** |

**两个深度模型各保留约 72–75% 的超随机效应在材质之外。**
这是"零样本视频检索不是材质解码换个包装"的正面证据，也是必须与头条数字一起报的那个数——
90 个同手同景视频让两者按构造重叠。

**必须排除单例材质（我第一版算错了，偏保守）**：每折 18 个 gallery 视频里有 2–3 个
**是其材质中唯一的一个**（实测 26/216 = 12% 的 gallery 槽位）。同材质内置换对它们是**恒等映射**——
保留的是**确切身份**而不只是材质，于是这些 query 在 null 里按模型真实精度得分，
把 null 抬高了一个**与材质知识无关**的量，使存活比例看起来偏小。
把单例目标从**两端**都剔除后：null 均值 0.0739/0.0749 → 0.0676/0.0682，
存活比例 63.3%/62.6% → **74.8%/72.0%**。
（`linear_align` 报的 38% 用的是未做该校正的旧约定，**不能与上表直接比**，须按同一约定重算。）

**同时纠正了一个会把这件事读反的表**：`cross_group` 的 gallery 是
{所有其它材质的项} + {真值}，真值是其材质中唯一的一个，所以**一个纯材质分类器在这上面拿 100%**
（合成分类器实测 cross_group top-1 = 1.000，而它的同材质 2-way 只有 0.500 = chance）。
它是被材质码**抬高**的上界，不是对材质的控制。`eval/report.py` 第 4 节此前把 within_group 与
cross_group 并排放在"属性捷径上限"标题下，已加显式说明并保留 cross_group 仅供对照。

---

# 二、六道门的实测证据

### G2 数据落地
raw BDF **80/80 = 107.7 GB**；derivatives 80/80 = 10.5 GB；events.tsv 80/80；
stimuli 384 mp4；VTD 90 行 / participants 80 行 / phenotype 4 表。

### G3 预处理
160/160 memmap（w0600 (2880,64,120) + wm100_800 (2880,64,180)），17.7 GB，
`baseline: null`、`n_dropped: 0`、逐被试 1:1 对齐、**单一 fingerprint** `7d219f48f433`。
- **EXG 定论：不存在**。BDF 头 65 通道 = A1–A32 + B1–B32 + Status。蓝图 §6.3「0 条 EOG」前提成立。
- 稳健尺度中位 17.3 µV（7.8–58.0），**0 个死通道**。
- 异常值记录：sub-17 缩放后 |x| 峰值 33298 σ（中位 45），sub-70 有 0.82% 样本超 20 σ。
- ICA 80/80：每人 2–5 个眼动成分（中位 3，**无人为 0**），
  代理相关中位 max|HEOG| = 0.86、max|VEOG| = 0.96 → F7–F8 / Fp1–Fp2 代理确实抓到了眼动。

### G4 视频编码器普查 — PASS，主终点维持视频级零样本
| 编码器 | 平均余弦 | PC top3 | 有效秩 | 坍缩 | LOVO top-5 | p |
|---|---|---|---|---|---|---|
| siglip2-base | 0.882 | 0.398 | 24.3 | no | 0.256 | 0.001 |
| clip-vit-l14 | 0.898 | 0.382 | 26.6 | no | 0.211 | 0.001 |
| xclip-base-32 | **0.579** | 0.396 | 23.7 | no | 0.200 | 0.001 |
| videomae-base-k400 | 0.920 | 0.353 | 26.4 | no | **0.278** | 0.001 |

（LOVO chance top-5 = 0.056。）RDM 与行为属性：四塔一致在 object（ρ 0.22–0.52）、
material（0.21–0.40）、valence（0.12–0.16）显著；videomae 另在 touch_type、approaching 显著。
**Q1a 用料**：朝向 vs 内容几何间隔 xclip **0.307** ≫ siglip2 0.101 ≫ videomae 0.054 →
**等变性分析应以 xclip 为主要对照塔**。
翻转轴实证确认：horflip = 左右、vertflip = 上下。

### G5 线性 MVPA — PASS（管线正确性），并纠正了一个会误导人的记分方式
`--cv sequence`，80 被试，聚类置换：

| target | 一致 chance | **多数类率** | 平衡精度峰 | 峰 ms | 已发表峰 | 判定 |
|---|---|---|---|---|---|---|
| orientation | 0.250 | 0.250 | **0.298** | **100** | 120–130 | **OK** |
| material | 0.125 | **0.311** | 0.129 | 250 | 110–120（起始）| 微弱 |
| toucher | 0.500 | **0.689** | 0.502 | 220 | — | 微弱 |
| touch_type | 0.083 | **0.356** | 0.084 | 285 | 165（起始）| 微弱 |
| valence | 0 | — | r=0.022 | **320** | 300 | 峰吻合 |

**管线正确性确认**：orientation 的刺激前基线 0.2483 恰在 chance 0.250 上、峰 100 ms；
valence t0 = −0.011 恰在 0 上、峰 320 ms —— 两个有干净基线的目标**都复现已发表时间课程**（±40 ms 内）。

**顺带纠正的记分陷阱**：用**朴素精度**记分时 material/toucher/touch_type 在 t=0 ms
就已 2.4×–4.6× 于「chance」（0.2996 / 0.6856 / 0.3413），看起来像强解码甚至像泄漏。
实测这三条曲线**恰好停在各自的多数类率上**（0.311 / 0.689 / 0.356）——
ds005662 的属性极度不平衡（skin 31%、object 69%、touch 36%），
**一个只会预测多数类的解码器就能拿到这个分数**，而 `1/n_classes` 的一致 chance 让它显得远高于随机。
orientation 是唯一平衡的标签（0.2504），所以只有它看起来正常。
已给 `linear_mvpa` 报告加 `majority rate` / `t0` / `evoked frac` 三列与
`NULL (majority class)` 判定，并把默认判据改成**经验地板**而非一致 chance。
换成平衡精度后的诚实结论：**只有 orientation 稳健解码（+4.8 pp）**，
material/toucher/touch_type 仅 +0.1~+0.4 pp。这与 companion 论文的口径差异要在文里写清。

### G6 主终点推断 — PASS（见 §一）
`$TACTUS_WORK/results/report_{infonce,protonce}/REPORT.md`：MDD 表印在第 1 节所有消融数字之前，
第 11 节列出「本设计无法证实的声明」（眼动限制在正文、属性簇不可分、无解剖定位）。

### Phase-0 审计
- **审计 A = INTERLEAVED —— 蓝图头号统计风险解除**。纯朝向序列占比 **0.000**，
  主导朝向 0.297（≈0.25），Cramér's V 0.092，互信息 0.009。§6.1-1 预案**不启动**。
- 80/80 design_complete，SOA 中位 0.7998 s，`onset_index_base` 一致投 0（残差精确 0.0）。
- **审计 C**：23 对属性关联 >0.5，`toucher↔object = 1.000`、`toucher↔material = 1.000`、
  `object↔material = 0.993` —— 在这 90 个刺激上**是同一个变量**；`threat↔pain = 0.971`。
  material×touch_type 空格 61/96。该 caveat 已硬编码进 census 报告与 REPORT 的"不可证实"清单。
- **material 分层算术与蓝图不符**：`{skin:28, metal:27, plastic:13, wood:10, cotton:4, fabric:3, sponge:3, hair:2}`
  （蓝图写「8 类 × ~11」）。5 折下每折 test 只覆盖 6–7/8 类；`splits.py` 优雅降级。
- 划分无泄漏：within_subject 5 / loso 8 / double_disjoint **40** 折。
  **D1 代价实测**：`adjacency_side="train"` 砍 34% 训练试次、**0 条**测试试次。
- 表型 n=80 无缺失：VT 1.84±2.97（**零膨胀**），EQ 16.06±5.14，IRI 24.82±4.35，
  MTS 自报 Yes **17/80**（蓝图预期 1–2 真阳性，须如实写）。三问卷与年龄无关（|r|≤0.124）。

---

# 三、发现并修复的问题

| 位置 | 问题 | 处置 |
|---|---|---|
| `losses/protonce.py` + config | **主目标存在可解捷径**：`live_positive=True`（出厂默认）下正样本 logit 用**当前可微**的 `<z_eeg,z_vid>`，负样本全是**滞后已 detach** 的 EMA 原型 → 视频投影头只要每步转离自己的滞后副本就能赢，**不需要任何 EEG–视频关系**。首跑即 `condition_acc=0.9999` / 验证停在 chance / 损失 1.159→0.003 | 受控实验证实（EEG 换纯噪声：True→100%，False→chance 0.018）。config 钉死 `live_positive: false`，构造函数在危险组合发 `RuntimeWarning`，噪声探针入 `tests/test_protonce_shortcut.py`。修复后损失 14.75→13.96 缓降、验证单调升 |
| `models/heads.py` | `TimeWindowHeads.forward` 在 `subject_context()` **之外**跑逐窗 head，而 `__init__` 恰把 SuLoRA adapter 挂在这些 head 上 → 该臂被试条件化恒为 0 | head 移入 context 内；自检 33/33 |
| `train/trainer.py` | **D2**：`n_subjects+1` 造出永不被训练的 row 0，未见被试靠 `strict_unseen=True` 侥幸绕开 —— 跑 transductive 变体即静默失效 | 去 `+1`；`UNSEEN_SUBJECT_INDEX` 0 → **−1**，走计算出的规则。日志实证 `rule='mean_of_train_subject_tokens', fixed_a_priori=True` |
| `train/trainer.py` | **ProtoNCE 生命周期未接线**（MISSION 遗留 1） | `_reset_prototype_bank()`（每折边界）、`_install_zeroshot_prototypes()`（评估前装留出 gallery，实测 coverage 0.822 = 训练 224 + 测试 72 / 360）、`_sync_prototype_bank()`（DDP） |
| `eval/probes.py` | **D6**：`("Fp","AF","F")` 前缀会连 FC*/FT*/Fz 一起删 —— 实测**删掉 26/64 通道（41% montage）**，把"是不是眼动"变成"是不是额叶" | 新增 `OCULAR_ABLATION_CHANNELS` 显式 8 通道白名单并设为默认 |
| `baselines/linear_mvpa.py` | 朴素精度对不平衡标签用 `1/n_classes` 当 chance → 多数类基线看起来像 2.4–4.6× 解码 | 加 `majority_rate` / `evoked frac` / `NULL (majority class)` 判定，判据改用经验地板；平衡精度分支单独出报告 |
| `baselines/linear_mvpa.py` | 组水平聚类置换要求 ≥3 被试，无法按被试分片 | 新增 `--no-group`（只缓存逐被试曲线），组检验交给汇总作业 |
| `models/video/encode.py` | transformers 5.x 起 `get_*_features` 返回 `BaseModelOutputWithPooling` 而非张量 → 3/4 编码器崩 | `_as_feature_tensor()` 统一解包 |
| `data/preprocess.py` | autoreject 硬编码 `n_jobs=1`（单被试 >15 min → 80 人 >20 h），且它正确地不在 fingerprint 里，导致"先 epochs 再补 autoreject"被静默跳过 | 拆出独立 `--stage autoreject`（读 memmap，不重读 1.3 GB BDF）+ `n_jobs` 接线 |
| `data/download.py` | `--what auto --subject-index` 被 submit.py 调用但**实现不存在**；80 并发写同一 JSON 必损坏；aws 配置多进程互踩；META 校验断言 80 份 per-subject sidecar + channels.tsv，但 ds005662 走 **BIDS 继承**（1 份顶层 sidecar、全树无 channels.tsv）→ 完整同步被误判失败 | 全部补齐/改正 |
| `slurm/submit.py` | 7 个 stage 的命令行**全部与真实模块 argparse 对不上** | 逐条改正 + 多分区列表 + `check_walltime()` |
| `configs/*` `Makefile` | `model_tag` 与产物名不符、`subject_conditioning` 含不存在的 `film`、`patch_len` 是 ATM 不存在的参数、`EMBED_CMD` 指向不存在的模块 | 全部对齐实现 |

### 新增基础设施
- **`slurm/pool.py`** —— 本集群 `QOSMaxSubmitJobPerUserLimit = 30`，**按 array 元素计**
  （实测 1-29 通过 / 1-30 拒绝），蓝图假定的 80 路 preprocess array 与 40 折训练 array **根本提交不了**。
  改为 W 个 worker 轮询共享 claim 目录消费 N 个任务：队列占用恒定、自动负载均衡、被抢占的 worker 重新入池。
  claim 用 `O_CREAT|O_EXCL`；**心跳写在文件内容里而非 mtime** —— 该文件系统对刚创建的文件返回
  `st_mtime == 0`，用 mtime 判定会让每个 worker 都以为 1 秒前的 claim 已陈旧 53 年并抢走它。
- **`tactus/eval/census.py`** —— G4 三判读，置换单元一律源视频，报告强制携带审计 C 的共线性 caveat。
- **`tactus/eval/run_report.py`** —— G6 驱动（`report.py` 只有渲染没有驱动）：
  收集逐被试检索 → 视频级置换 + 试次级窄化对照 → 分半噪声上限 → `REPORT.md`。
- **`slurm/setup_env.sh`** —— conda 环境 `tactus`（py3.11 / torch 2.6+cu124 / mne 1.11 / autoreject / picard / transformers 5）。

### 集群事实（已写进 `slurm/cluster.conf`）
- 无 slurmdbd → 不需要 `--account/--qos`
- **3090 限 4 CPU/GPU，且 slurm 按多分区列表中最严的校验** → GPU 作业一律 `--cpus-per-task=4`
- GPU 多分区调度：`V100,P100,A30,A40,3090,L40S,A100`。实测起始 V100/P100/A30 立即、
  A40 +4.5 h、A100 +9 h、H100 +12 h、**L40S +27 h** —— 默认排 L40S/H100 会白等一整天
- `/projects/EEG-foundation-model` 98% 满，余 659 GB；本项目占约 128 GB（数据 107 + 派生 21）

---

## DECISIONS_NEEDED
- **none**。MISSION 预留给人的三件事全部有确定答案：
  审计 A = **INTERLEAVED**；EXG 通道 = **不存在**（BDF 头实查）；G4 三判读 = **全部通过**，主终点不改。

## 下一步
1. **换对比学习算法**（本仓库的设计目的）：`tactus/losses/my_loss.py` + `losses/__init__.py` 一行 import
   + `configs/my_loss.yaml` 一个 `loss.name` 键，然后
   `python slurm/pool.py submit --name train_myloss --tasks 0-4 --workers 5 --gpus 1 ...`。
   基线已就位：InfoNCE 10.99% / ProtoNCE 11.93%，配伪试次、置换 p、上限分数、MDD。
2. G6 剩下的眼动判据：用修好的 `OCULAR_ABLATION_CHANNELS` 跑前部消融 + EOG 代理基线对照，
   只在前扫视窗（<150 ms）作"非眼动"声明。
3. `double_disjoint` 40 折主张格（每折约 50 min，5 worker 约 7 h）。
4. autoreject 跑完后把伪迹率并进 Q3 的 SNR 协变量集。

---

# 四、第二轮：从未执行过的基线模块审计（8 agent，含对抗性验证）

MISSION §5.3 的基线阶梯有 6 级，此前只跑了第 1 级和第 4 级的一半。对 4 个**从未执行过**的模块
做了并行冒烟 + 独立对抗性验证（每个 audit agent 配一个专门找漏的 verifier）。
**结果：30 个 bug，4 个模块全部能跑，但 verifier 推翻了 4 份报告中的 3 份**——
audit agent 说"能跑且数字合理"，verifier 证明数字本身测的是错的东西。

| 模块 | 能跑 | verifier 判定 | 关键问题 |
|---|---|---|---|
| `corrca.py` | ✅ | **推翻**（数字全部可复现，科学声明不成立） | 见下 |
| `srm.py` | ✅ | **推翻**（无泄漏，但配置污染 + SRM 步骤在毁信号） | 见下 |
| `linear_align.py` | ✅ | **未被推翻** | 唯一经受住对抗验证的模块 |
| `probes.py` 眼动半边 | ✅ | **推翻**（结论是 bug 制造出来的） | 见下 |

### 4.1 CorrCA：头条数字测的是刺激无关的诱发响应
- 把每个被试的 360 个条件平均**独立置换**（销毁全部刺激对应关系，其余不变），
  c1 仍保留 **91.4%**（0.1522 → 0.1392）。分解证实：只在总平均 ERP 上重拟合得 0.6221，
  只在条件特异残差上重拟合得 0.0191。**报告的 0.1522 是所有 360 条件共有的刺激 onset 响应；
  真正刺激特异的 ISC 约 0.019，小 8 倍。**
- 原先的 null 是**错的 null**：白噪声不检验任何时间锁定成分；逐被试环形时移恰好把 onset ERP 去相位，
  即抹掉了主导统计量的那个成分。对正确的刺激身份 null，margin 是 **1.09×**，不是 15×。
- **sub-17 一个人贡献了 trace(R_w) 的 90.9%**，这个"80 被试"拟合的 Kish 有效样本量 = **1.2 个被试**。
  去掉 sub-17，c1 +34%、c2 +50%、c3 +73%，已发表的 comp-1 滤波器与无 sub-17 版本只相关 0.888。
- `subject_isc` **不是尺度不变的**：把一个被试整体乘以常数，ISC 从 0.259 变成 0.046（×10）或 0.135（×0.25），
  而真值恒为 0.259。这直接摧毁它作为 Q3 SNR 协变量的用途——**对增益/阻抗敏感的回归量本身就是"帽子贴合"声明**。
- **已修复并重跑**：默认移除总平均 ERP、默认报告条件置换 null、逐被试幅度归一化、改报尺度不变 ISC。
  头条 per-pair c1 **0.1522 → 0.0325（降 4.7 倍）**，对正确 null 的 margin **1.09× → 3.50×**，
  sub-17 份额 **90.9% → 1.25%**，滤波器稳定性 0.888 → 0.995。修完后 c4/c5 不过自己的 null（1.32×/1.27×），
  只有 c1–c3 成立。**新旧 SNR 回归量的 Spearman 只有 0.610——已用旧列做过的表型分析必须重做。**

### 4.2 SRM：SRM 这一步在毁信号
- 同一折、同一 query/gallery，只换对齐步骤：无-SRM 对照 72-way top1 = **0.0278**（2.0× chance），
  模块的 SRM = **0.0097（低于 chance）**。k=5/10/20/40 全部 ≤ chance。损失单调收敛（1.87% 方差下降），
  DetSRM 代数经检查正确。数据里**确实有**共享信号：半-半组平均在同样的留出条件上检索到 72-way 0.0278 / 18-way 0.1069。
  诊断：特征是 channel×time 且**无逐特征标准化**，Frobenius 目标被高方差漂移/alpha 成分主导，
  f=3840 ≫ n=288 时 k=20 的共享空间只吃到 <2% 方差。
  **没有这个无-SRM 对照就发表"SRM ≈ chance"，会被审稿人当场打掉。**
- 目录键 `{regime}_{window}_k{k}_{feature_mode}` **不含** subjects/seed/decim/n-iter/model-tag 等，
  实测一次全新 80 被试运行落进了 6 被试旧结果的目录、只打印 "cached"、并给出**不同的** enrollment 曲线；
  summary.json 仍标 `stimulus_disjoint: true`。
- `--regime loso` 无 `--allow-stimulus-overlap` 时**退出码 0 但什么都不写**，调度器会记成功。

### 4.3 眼动 driver：G6 判据 2 的"无信号"结论是 bug 造出来的
- gallery 用了**全部 90 个视频**而非该折的 **18 个留出视频**，其中约 14 个是岭回归训练过的；
  且只评分含截距的预测，而截距项在这个各向异性 SigLIP gallery（平均成对余弦 0.88）上
  **把 18-way top-1 钉死在 chance**（`linear_align.py` 早就把这个失败模式写进注释并同时评分两个变体）。
  结果每个 arm 都停在 0.0077–0.0098，比 chance 低 5.7 倍，模块据此打印 "UNINFORMATIVE，没有信号可消融"。
- 改用该折 18 个留出视频 + 去截距变体后（8 被试、fold 0、同折同探针）：
  `full_eeg 0.0840 / ocular_ablated 0.0833 / ocular_surrogate 0.0764`，
  单试次 0.07579 vs chance 0.05556（n=4341，**约 5.8 sigma**）。
  **G6 判据 2 是可回答的**：信号存在、消融后存活、EEG 超过 EOG 替代基线。修复后正在按规模重跑。

# 五、我自己发现并修掉的两个真 bug

### 5.1 CompositeLoss 的 warmup 计数器从未前进 → **λ₁ 恒等于 0**
`CompositeLoss.step()` 存在，但 **trainer 从来不调用它**（只调 `scaler.step` / `sched.step`）。
`_step` 永远是 0，于是任何带 `warmup_steps` 的组件 `effective_weight = w × (0-0)/warm = 0.0`。
实测 `atm_composite.yaml` 的 CLISA（weight 0.2, warmup_steps 500）在**每个 epoch 都记录 weight = 0.0**。
而这个 config 的存在理由**正是**跑 λ₁ 冗余消融（蓝图 §4.3）——λ₁ 恒为 0 时它什么也没测。
已修；ramp 实测 0 → 0.1（250 步）→ 0.2（500 步）。原先跑出的 ATM 结果已删除并按
**λ₁=0.2 / λ₁=0 两格**重新提交（顺带把消融做成真正的对照）。
这正是 MISSION §3 警告的"悄悄贡献 0 的损失项，日志上看起来和正常工作的一模一样"。

### 5.2 双重留出 40 折的**折序陷阱**
`make_folds("double_disjoint")` 按 video-fold-major 发出（`for v in range(5): for s in range(8)`），
所以 fold 0..7 **留出的是同一批 18 个视频**。我原计划的 Phase-2 试点 `--tasks 0-4`
会**把同一个视频折评 5 遍**，刺激方差采样为零——而刺激泛化声明恰恰建立在它上面。
失败是静默的：每折单独都合法，被试数看起来是 40，只有视频级标准误悄悄为 0。
已加 `splits.fold_run_order()`（subject-fold-major，任何前缀都跨视频折）并测通。
本次 40 折是**全跑**，所以结果不受影响。

# 六、数据质量：sub-17 的坏电极

- **实测全 80 被试**：sub-17 的 PO4 通道 sd 是该被试通道中位数的 **80 倍**（次差 sub-54 的 F8 只有 9.5 倍），
  max|x| = 33298 稳健 sigma（cohort 中位 ~45）。其余 15 个被标记的被试都是额部（AF7/Fp2/Fpz/AF8），
  即眼动/EMG，属正常且已由眼动 battery 处理。**只有 sub-17 是真正的坏电极。**
- **为什么预处理没抓到**：逐通道稳健缩放用的是 IQR，而 **IQR 按构造对尖峰免疫**——
  坏电极的 IQR 完全正常（sub-17 PO4 的 robust_sd_ratio = 1.00），只有**普通 sd 比值**看得见它（82.7）。
  任何以稳健离散度为准的 QC 都会再漏一次。
- **预处理其实已经算出了铁证然后扔掉**：每个 sidecar 都写了 `abs_max_after_scaling`（sub-17: 33297.9）
  和 `frac_abs_gt_20`，但没有任何代码读它们。**已补上写入时的 QC 闸门**（`SD_RATIO_WARN` / `ABS_MAX_WARN`），
  实测对 sub-01 静默、对 sub-17 触发。
- **对已报告的深度模型结果无影响**（我直接验证）：sub-17 的主终点 = 0.0997（z = −0.69），
  连最低 5 名都进不去；去掉它 cohort 均值只动 **+0.025 个百分点**。
  原因是 trainer 的 `clamp=20` 保护了它，而 corrca 直接读 memmap 所以中招。
- 一个真实但无害的机制：EA 白化在 clamp **之前**拟合，所以 sub-17 的白化基底确实被 PO4 尖峰污染——
  实测 corr(W_raw, W_clamped) = 0.948 但 ‖ΔW‖/‖W‖ = **9.96**（干净被试 0.0006）。
  后续的逐通道稳健重缩放把幅度失真抵消掉了，这就是它仍能正常得分的原因。
- **20 sigma clamp 对 sub-17 不是无害修复**：clamp 前后条件平均图样只相关 r = 0.162，
  因为该图样 99.75% 的能量是伪迹。**"clamp sub-17" 与"排除 sub-17"是两个不同决定，需要人拍板。**

---

# 七、基线阶梯第 5 级：EEG 基础模型微调（已实现，运行中）

蓝图把这一级定位为**指示性基线、只跑 3 折**，并要求把 patch 长度与我们 600 ms epoch 的
**接口错配写成文**。三个模型全部接上了，**权重都是真的**（独立验证器逐张量比对）：

| 模型 | 来源 | 预训练比例 | 参数 | 窗口 / patch | **padding 比例** | 通道映射 |
|---|---|---|---|---|---|---|
| **LaBraM**-base | `braindecode/labram-pretrained` | **221/221 张量，5,817,136/5,817,136 = 100%** | 5.8M | w0600，patch 200 不变 | **40%**（replicate, center） | 64/64 名称命中 `LABRAM_CHANNEL_ORDER`，无插值 |
| **CBraMod** | `braindecode/cbramod-pretrained` | **211/211 张量，4,924,000/4,924,000 = 100%** | 5.7M（读出后 9.24M 可训练） | w0600，patch 200 不变 | **40%**（reflect, right） | 通道无关（ACPE），但通道**顺序**与 TUEG 10-20 不同，未量化 |
| **EEGPT** | `braindecode/eegpt-pretrained` | **103/103 张量，25,287,168/25,287,168 = 100%** | 25.3M | w0600，patch 64/stride 32 不变 | **6.25%**（replicate, left） | **P9/P10/Iz/AFz 四个电极实质被丢弃**（EEGPT 无对应通道嵌入） |

**做法与证据**：`braindecode 1.7.0` 同时提供三个架构与 HF 上的预训练权重，省掉重实现三篇论文。
每个 wrapper 都不用 `from_pretrained`（它 `strict=False` 会**静默跳过**不匹配张量），
而是逐 key 拷贝 + 形状检查 + 载入后 `torch.equal` 复核 + 低于阈值直接 **raise**。
对照实验：随机初始化的 CBraMod 只与 checkpoint 匹配 1/211 张量，所以"全匹配"不是平凡结论。

**接口错配是这一级的核心结论，必须与数字一起报**：
- **LaBraM 的 patch 不能改小，而且后果比"丢掉 patch 嵌入"严重得多**。
  LaBraM 的 patcher 是全卷积的，**它的输出宽度就是 transformer 的 embed_dim**，
  所以 `patch_size=120` → `embed_dim=120`，patcher 之后**每一个张量的形状都变了**。
  实测只剩 **12/221 张量、576/2,108,112 参数 = 0.03%** 可载入，
  而那 12 个恰好是三个 `_TemporalConv` 卷积加它们的 GroupNorm——**唯一与形状无关的层**。
  换句话说：**整个模型（transformer 在内）都是随机初始化的**，不只是 patch 嵌入。
  而 braindecode 在 `n_times=120` 时**只发一个 UserWarning** 就这么做了。
  wrapper 现在把这个 UserWarning 提升成 `RuntimeError`，`patch_size=120` 直接 raise。
  所以 padding 是唯一能保住"基础模型"这个词的做法。
- **CBraMod 在 pad_to=200 时只有 1 个 patch**，而它的招牌机制是 (通道 × patch) 的 criss-cross 注意力，
  时间那一路因此**退化**。这是验证器发现的一级缺陷。
- **EEGPT 不是"零 padding"**（我先前的侦察说法有误）：它要求 `(T−64)%32==0`，
  120 → 128 = 3 个 patch，padding 6.25%；braindecode 自己会插 `ConstantPad1d((0,8), 0)`，
  而我们的 epoch 没有基线校正（边缘均值 ≈ 0.53 稳健 sigma），零填充会伪造一个阶跃，
  所以 wrapper 先做 replicate 预填充让它变成 no-op。
- **EEGPT 还有采样率错配**：checkpoint 是 250 Hz，我们喂 200 Hz，每个 patch 实际是 320 ms 而非 256 ms。
- 三者的 `TimeWindowHeads` 子窗都会被 padding 主导（LaBraM/CBraMod 75–85%，EEGPT 的最小感受野
  320 ms 已宽于 0–150 ms 早窗）→ **这一级不得报时间分辨曲线**。

**为此改的框架代码（都在我自己的文件里）**：
- `trainer.py` 新增 **encoder 自定义参数组接线**：`encoder.param_groups(lr, wd)` 若存在则前置。
  预训练 backbone 与新初始化 projector 必须用不同 LR，而**只能靠 per-group `lr`**——
  AdamW 对梯度尺度不变，缩放梯度的 hook 完全无效。实测 per-group LR 能正确穿过 cosine 调度
  （0.5 lambda 后 1.5e-05 / 1.5e-04）。此前 `backbone_lr_scale` 是死键。
- `trainer.py` 把 `encoder.describe()` 打进 log.jsonl——预训练比例/padding 比例/通道映射
  **就是这一级数字的 caveat**，必须与数字同处一地，不能只躺在 docstring 里。
- `models/selftest.py` 新增 `--all-registered`：它原先把架构列表**硬编码**成 [tsconv, atm]，
  于是后加的编码器**永远不会被契约 F 检验**却仍显示全绿。
  现在走注册表，实测 **105/105 通过**（含三个 FM × 4 种被试条件化）。
- `models/eeg/__init__.py` 用 **try/except 守卫**注册 FM——它们依赖可选重型依赖 `braindecode`，
  而这个包在 trainer/selftest/所有基线的 import 路径上；缺依赖必须降级成"该编码器不可用"，
  不能变成"整个包 import 失败"。`FM_ENCODERS_AVAILABLE` 暴露每个的状态。
- 环境：`braindecode==1.7.0`；**注意顺序**——它会拉 `torchaudio` 最新版（2.11，链接 libcudart.so.13），
  与我们的 torch 2.6.0+cu124 冲突且 import 即死，必须随后重装 `torchaudio==2.6.0+cu124`。已写进 `setup_env.sh`。

**配置口径**：三个 config 统一 `regime: within_subject`、folds 0/1/2。
`cbramod_ft.yaml` 原本写 `double_disjoint` + "`--max-folds 3`"，那正是**同一视频折评三遍**的陷阱
（见 §五-5.2），已改并在注释里写明理由。

**尚未解决 / 必须随数字一起说的**：
- 超参未调（无 GPU 可扫）：LaBraM/EEGPT lr 1e-4、CBraMod 3e-5、30 epochs，都是猜的。
  若这一级表现差，**在成为"关于模型的结论"之前先是三重混淆**（padding、未调 LR、未调冻结策略）。
- 参数预算不对等：CBraMod 读出层 12800 宽 → 9.24M 可训练，而 tsconv 只有 0.33M。
- EEGPT 的幅度单位是**推断**的（模型卡未写），冻结特征探针在所有变体上都停在 chance。
- 3 折 = **无推断**：没有视频级置换 p、没有 fraction-of-ceiling。只是侦察信号。
