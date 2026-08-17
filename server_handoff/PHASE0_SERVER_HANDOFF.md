# TACTUS Phase 0 — Server 端执行手册

**用途**：本手册 + 同目录两个脚本交给 server 端（Linux）直接执行，完成 BLUEPRINT_v2.md §6.1/§8 Phase 0 的三大审计。全程不需要 EEG 数据本体——第 1–3 步只用元数据与刺激视频（<1 GB），第 4 步才是大文件下载（可选、可后台挂起）。

**执行者注意**：每步末尾有"预期输出"和"判读规则"。所有产物写入 `./phase0_out/`。遇到列名/路径与预期不符时，先打印实际结构再适配，不要跳过审计。

---

## 第 0 步：环境

```bash
conda create -n vtalign python=3.11 -y && conda activate vtalign
pip install numpy pandas scipy scikit-learn awscli opencv-python-headless torch transformers
```

GPU 非必需（第 3 步 CPU 约 10–20 分钟；有 GPU 则 ~2 分钟）。

## 第 1 步：元数据下载（~几十 MB，1–2 分钟）

OpenNeuro S3 公开桶，无需凭证：

```bash
mkdir -p ds005662 && aws s3 sync --no-sign-request s3://openneuro.org/ds005662 ds005662 \
  --exclude "*" \
  --include "*.tsv" --include "*.json" --include "CHANGES" --include "README*" \
  --include "code/analysis/*" --include "phenotype/*" --include "sourcedata/*.csv"
```

验证：应有 80 个 `ds005662/sub-XX/eeg/sub-XX_task-video_events.tsv`（各 ~850 KB）、`ds005662/code/analysis/VTD.csv`（90 行）、`ds005662/participants.tsv`、`ds005662/phenotype/{EQ,IRI,VT,MTS}_data.tsv`。

## 第 2 步：核心审计（audit_phase0.py，秒级）

```bash
python audit_phase0.py --bids ds005662 --out phase0_out
```

脚本做四件事（对应 BLUEPRINT_v2 §6.1 审计 1+2 及 §1.3 坑位核实）：

**审计 A（最高优先级）— 序列×朝向×源视频交叉表**。逐被试解析 events.tsv（`istarget==0`，从 `stim` 路径解析朝向 {original, horflip, vertflip, horvertflip} 与源视频号 1–90），统计每个 `sequencenumber` 内的朝向构成。
- **判读**：若绝大多数序列只含 1 种朝向 → **朝向按序列成块（BLOCKED）**，蓝图中朝向解码/等变性设计全部触发降级预案（§6.1-1）；若每序列 4 种朝向充分混合 → 解除警报。2880/32=90 的算术使 BLOCKED 是先验更可能的结果——无论哪种结果都直接写进论文。
- 附带输出：每序列 unique 视频数（=90 则"每序列过一遍全部视频"）；朝向×序列号 Cramér's V；同一序列位置的朝向是否跨被试一致（区分"被试内成块"与"设计层面固定成块"）。

**审计 B — 时间结构与污染核实**：SOA 分布（组内 onset 差分，预期紧贴 0.800 s——偏差大则 RSVP 时序假设要修）；目标试次及**目标后继试次**计数（后者带按键运动电位，训练必须剔除，脚本输出逐被试剔除清单规模）；`presentationnumber` 语义核实（是否 1–8 重复计数）；试次序号与标签的互信息（trial-index 泄漏的先验量级）。

**审计 C — 90×90 属性互相关矩阵**（VTD.csv）：连续对（valence/arousal/threat）用 |Pearson r|，类别对（material/touch_type/toucher/object/pain/approaching）用 Cramér's V，混合对用相关比 η。输出全矩阵 CSV + **关联 >0.5 的对子清单**（这些对子在 90 个刺激上不可分离，Q1b 禁答）+ material×touch_type 列联表及空格计数（属性级零样本分层可行性的直接证据）。

**审计 D — 表型表体检**：participants.tsv 的 VT/EQ/IRI/MTS 分布、缺失值、与年龄/性别的相关（Q3 协变量集的第一眼）。

**预期输出**：`phase0_out/audit_report.md`（含四个审计的判定行）、`seq_orient_crosstab.csv`、`attr_association_matrix.csv`、`per_subject_summary.csv`。

## 第 3 步：刺激下载 + 编码器普查（encoder_census.py）

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds005662/code/experiment_files/stimuli stimuli
```

```bash
python encoder_census.py --stim stimuli --vtd ds005662/code/analysis/VTD.csv --out phase0_out
```

对 90 个原始朝向 mp4（`videos_short_600ms/`，排除 flip 目录与 target 目录）做三项测试（BLUEPRINT_v2 §4.1）：

1. **嵌入坍缩检查**：SigLIP2 逐帧嵌入均值池化 → 90×D 矩阵；报告平均成对余弦相似度与 PC 谱。判读：平均余弦 >0.9 且前 3 PC 解释 >80% → 坍缩警报，触发备胎（触觉语义族/中层特征）。
2. **RDM–行为对齐**：嵌入 RDM（1−cos）vs 属性 RDM（valence/arousal/threat 连续差 + material/touch_type 失配），Spearman + 视频级置换 p（1000 次）。判读：≥1 个属性 RDM 显著相关 → §8 Phase 0 go 条件之一满足。
3. **属性→嵌入可预测性（决定主终点的测试）**：leave-one-video-out 岭回归，用属性预测被留出视频的嵌入，按余弦在 90 个真嵌入中检索，报告 top-1/top-5/平均秩 + 属性置换零分布。判读：**top-5 不显著高于置换 → 视频级零样本在原理上不可行，主终点改为属性级泛化**（蓝图预案，不是失败）。

**预期输出**：`census_report.md`、`embeddings_siglip2.npy`、`rdm_embed.csv`。若要多编码器对比，换 `--model` 重跑（默认 `google/siglip2-base-patch16-224`；可试 `google/siglip-so400m-patch14-384` 等，脚本按模型名分文件保存）。

## 第 4 步（可选，后台挂起）：大文件

预处理 epochs（~11 GB，MVPA 复现用）：

```bash
aws s3 sync --no-sign-request s3://openneuro.org/ds005662/derivatives derivatives
```

原始 BDF（~110 GB，正式管线用——v2 已确认是硬依赖，早下早好）：

```bash
nohup aws s3 sync --no-sign-request s3://openneuro.org/ds005662 ds005662_full > sync.log 2>&1 &
```

## 回传物

`phase0_out/` 整目录 + 一段结论：审计 A 的 BLOCKED/INTERLEAVED 判定、审计 C 的禁答对子清单、普查三判读（坍缩？RDM 对齐？属性可预测嵌入？）。这五个答案决定 BLUEPRINT_v2 §8 的 go/no-go 与 Q1/主终点的最终形态。
