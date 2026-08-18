# TACTUS 蓝图 v3：分解式多正样本 EEG–视频对比学习（合并版）

**版本**: v3 终稿（2026-08-18）。由两条独立路线合并而成：v2（本项目蓝图，经统计/ML/定位三路对抗审查）+ 外部 design.md（独立分析，贡献了 factorized 框架与关键文献）。合并裁决与取舍见 §11。**本文件取代 v2**；v2 的统计与混淆层全部继承，此处只写结论。
**数据集**: ds005662 v2.0.1（OpenNeuro，CC0）——数据档案、文件坑位、companion 论文基线详见 v2 §1（未变，不重复）。
**代号**: **TACTUS** = Touch Alignment by Contrastive Transfer to Unseen Subjects
**代码**: `tactus/` 仓库已实现本蓝图（`configs/factorized_fhmc.yaml` 为旗舰配置；FHMC 损失 `tactus/losses/factorized.py` 已过 8 场景对抗自检 + 全家族 80/80 回归）

---

## 0. 定位（v3 重写）

### 核心问题：竞争性不变性（competing invariances）

数据集的 4 种朝向翻转具有双重身份：既是**同一触觉内容的增强**（划、捏、敲的内容不因镜像而变），又携带**真实的视角/利手信息**（垂直翻转操作化 self/other，水平翻转操作化左右手，且 companion 论文证明两者都可从 EEG 解码）。单一共享嵌入空间被迫二选一：把翻转当正样本 → 删除视角信息；当负样本 → 强迫编码器从 EEG 判别镜像，容量倒向低层视觉/眼动轴。**这个结构性冲突在静态图像基准（THINGS-EEG）中不存在，是本数据集独有的建模问题，也是论文的方法学引擎。**

### 方案：FHMC（Factorized Hierarchical Multi-positive Contrastive）

主干空间保留 **exact** 条件级对齐（主终点在此测量，不动），三个分解头分别学习：

| 头 | 目标 | 正样本定义 |
|---|---|---|
| content | 翻转不变的触觉内容 | 同 `video_id`（跨朝向、跨重复、跨被试）|
| geometry | 翻转等变的视角编码 | 同 `orientation`（跨视频）+ 4-way 朝向监督 |
| semantic | 情感语义连续结构 | 软目标 `w_ij = exp(-(|Δv|+|Δa|+|Δt|)/σ)` |

外加 content⊥geometry 交叉协方差解纠缠正则。多正样本结构由数据生成机制直接决定（80 被试 × 8 重复 × 4 朝向共看同 90 事件），不是可选的技术修饰。

### 贡献排序（论文骨架）

1. **FHMC 框架**：分解式多正样本对比目标，架构级解决竞争性不变性冲突；factorization 由线性探针表验证（content 头读出视频/材质高、朝向低；geometry 头相反；两头被试身份都低）
2. **双重留出泛化**：未见被试 × 未见视频的零样本对齐 + 跨被试缩放曲线（79 人 LOSO vs 文献 9–19 人），由 80×360×8 稠密共享设计独家支撑；跨被试正样本是**直接观测的**而非对抗式诱导的（与 SAM-Net 的本质区别）
3. **时间分辨对齐 + 等变性几何**：分窗对齐起始曲线（与 companion MVPA 延迟直接可比）；嵌入在刺激翻转下的不变/等变子空间分解
4. **支撑终点**：表型失谐设计（v2 §7 协议原封不动：分离编码器、SNR 偏相关、双重分离才可声明）

明确不做：视频重建（90 同域刺激正中 spurious-reconstruction 批评）；TPJ/pSTS vs S1 解剖裁决（传感器空间过度延伸）；任何"首个 EEG-视频对比学习"措辞（EEGMirror 已存在，见 §2）。

---

## 1. 数据集档案

见 v2 §1，全部继承。工程要点重述最关键三条：主训练窗 **0–600 ms 无基线校正**（−100~0 基线泡在上一试次晚期情感响应里，800 ms SOA 所致）；**原始 BDF 是硬依赖**（0 条 EOG 通道，ICA 需连续数据）；目标后继试次永久剔除（按键运动电位）。

---

## 2. 竞争格局（v3 更新：EEG-视频不再是空白，收窄措辞）

**新核实的直接相关工作**（外部 design.md 贡献、已逐条验证为真）：

| 工作 | venue | 与我们的关系 |
|---|---|---|
| **EEGMirror**（Liu et al.）| **ICCV 2025** | EEG-视频多模态对比对齐 + 重建，montage-agnostic 自监督。**否决任何"首个 EEG-视频对比"措辞**。但它是重建中心、通用视频、无受控因子结构 |
| **SAM-Net**（Han et al.）| **CVPR 2026** | 跨被试 EEG-视频重建，subject-adversarial。跨被试也非空白；我们的差异 = 跨被试正样本直接观测（80 人共看同刺激），非对抗诱导 |
| NEED | OpenReview | 跨被试跨任务视频/图像重建 |
| MUSE（similarity-keeping）、MB2C（cycle consistency）、CI-BVCL（因果混杂）、D²-FOSA（CVPR 2026）| 各处 | 结构约束对比学习的相邻方法；MUSE 进基线阵容 |
| CET-MAE（ACL 2024）| — | 教训入库：纯跨模态对齐会掏空 EEG 编码器内部结构，配 masked-EEG 重建辅助项（已列为可选损失）|

**站得住的新颖性声明**（一句话版）：
> 受控动态触觉刺激下，利用直接观测的跨被试多正样本结构，分解翻转不变内容与翻转等变视角，并在未见被试 × 未见刺激的双重留出下评估。

THINGS-EEG 社区（NICE→ATM→2026 wave）、作者组（线性 MVPA + 个体差异意向）、替代性触觉 ERP 社区的三方格局与时间窗判断不变（v2 §2）；ds005662 至今外部零引用，速度仍然要紧。

---

## 3. 科学问题（v3 定稿）

**Q1（因子化，架构实现）**：观看触摸的神经编码能否分解为翻转不变内容与翻转等变几何？FHMC 的 content/geometry 头就是这个问题的可训练形式；等变性几何分析（翻转作用下的子空间变换）是其模型无关的验证。**前提门**：审计 A（序列×朝向交叉表）+ 眼动 battery 通过后 geometry 头结论才可报告（§6）。属性级方差分割维持 v2 的"有界描述性"定位（90 片段属性相关结构先行发表）。

**Q2（特征层级时间课程）**：不变（v2）。分窗对齐头（0–150/150–350/350–600 ms）产出对齐起始曲线；预期几何轴早（~60–130 ms）、内容/语义轴晚（130–300 ms+）——FHMC 的分解头让这条曲线可以**按因子分别画**，这是单一嵌入做不到的。

**Q3（表型失谐）**：不变（v2 §7 全套：λ₁=0 无条件化分离编码器、噪声上限/伪迹率/注意力/年龄性别偏相关、情感轴预测 VT ∧ 几何轴不预测 ∧ SNR 不解释的双重分离、10 探测视频被试内检验、MTS 仅个案）。FHMC 附赠一个更锐的版本：**per-subject 的 semantic 头对齐强度 vs geometry 头对齐强度**天然构成失谐对——同一被试内的因子对比自动控制了全局 SNR。

---

## 4. 模型与目标函数（v3 定稿）

### 4.1 总损失

```
L = L_exact(主干)                       # 条件级多正样本 InfoNCE，主终点空间
  + λc·L_content + λg·L_geometry        # 分解头（含 4-way 朝向 CE）
  + λs·L_semantic                       # 情感核软目标
  + λd·L_disent                         # content⊥geometry 交叉协方差
  [+ λm·L_maskedEEG]                    # 可选：CET-MAE 式重建辅助（防编码器坍缩成投影器）
  [+ λ₁·L_CLISA]                        # 可选：跨被试项（与 exact 部分冗余，单独消融）
```

已实现：`tactus/losses/factorized.py`（头在损失模块内——trainer 已优化并 checkpoint `loss_fn` 参数，零改动接入；`embed_content/geometry/semantic()` 方法供评测接线）。**朝向政策的分叉就此消解**：不再是"翻转当正样本还是负样本"的二选一试点，content 头收正样本、geometry 头收监督信号，两者并存。v2 的两格试点降级为消融行。

### 4.2 批次构成（正确性问题）

FHMC 必须配 `batch.mode: video_x_subject`（24 视频 × 3 被试）：content/geometry 项只有在批内出现重复视频/朝向时才有多正样本结构；`distinct_video` 会让 content 项静默塌缩为普通 InfoNCE。

### 4.3 视频侧与文本侧

编码器普查、坍缩检查、属性→嵌入可预测性测试维持 v2 §4.1（Phase 0 强制）。文本锚改用**结构化确定性 caption**（从 VTD 属性模板生成："A third-person view of a left hand being slowly stroked with a sponge; pleasant, low threat"），不用自由 VLM caption——可控、可归因到具体因子。350 评分者分布仍是 SoftCLIP 软目标与分歧度权重的来源（v2 §4.4）。

### 4.4 EEG 侧与基线阵容

EEG 编码器、被试条件化三选一、EA、未见被试规则（先验固定）全部不变（v2 §4.2 + MISSION D2）。基线阵容合并双方：companion 线性 MVPA（管线总检验）、EA+线性、CorrCA/SRM、NICE-InfoNCE、**ProtoNCE**（降为强基线臂）、**MUSE 式 similarity-keeping**（新增）、NICE-LLM 式文本增强、EEG FM 全微调（3 折指示性）、FHMC 完整 + 逐项消融（λc/λg/λs/λd 置零）。

---

## 5. 评测协议

v2 §5 全部继承（划分、置换单元=源视频、MDD 前置声明、噪声上限、5×8=40 双重留出格、主终点 `test/video/g18/top1_pseudo`）。v3 新增三项：

1. **因子化探针表**（FHMC 的验收标准）：content 头 {video 90-way, material 8-way} 高 / {orientation} 低；geometry 头相反；两头 {subject ID} 都低且与对齐保留率联合报告。这张表是"factorization 成立"的证据，缺它就只是多头检索
2. **动态性对照阶梯**（design.md 贡献）：center-frame / average-frame / **shuffled-frame** / true-order 视频编码器四级——只有 true-order > shuffled 才能声明模型使用了触摸动作的时间结构（`encode.py` 加 `--frame-order` 旗标，见 MISSION 待办）
3. **检索 gallery 双层报告**：90-way 源视频级（忽略朝向）与 360-way 精确刺激级并报；gallery 内每个候选只出现一次

---

## 6. 混淆控制

v2 §6 全部继承（Phase 0 三大审计、0–600 ms 无基线主管线、EOG 替代与前扫视窗声明、trial-index 对照、watermelon 置换）。v3 强化一条：

**Geometry 头的双重门（D10）**：geometry 头恰好建在最早最强、也最可能被污染的信号上——(a) 若审计 A 判定朝向按序列成块，geometry 头部分是时段解码器；(b) 镜像翻转系统性镜像注视模式，geometry 头可能是眼动读出。因此 geometry 相关的一切结论以**审计 A 判定 + 眼动 battery（EOG 替代基线被超过、前部消融存活、前扫视窗）**双重通过为前提；未通过则 geometry 头保留为架构组件但结论降级为"含未定成分的视角相关信号"。

---

## 7. 表型协议

v2 §7 原封不动，加 §3 所述的因子内失谐对（semantic 头对齐 vs geometry 头对齐的被试内对比）作为首选操作化。

## 8. 路线图

阶段与验收门维持 MISSION.md S0–S7/G0–G6。变化：S6 从两臂改**三臂**（`nice_infonce` 参考锚 / `nice_protonce` 强基线 / `factorized_fhmc` 旗舰）；Phase 0 普查追加逐编码器帧数 padding 记录（15 帧 vs V-JEPA2 的 64、VideoMAE 的 16）。投稿目标不变：ICML 2027 主选，CCN 摘要 + Phase 1 arXiv 占位链。

## 9. 计算预算

v2 §9 不变。FHMC 增量可忽略（57k 损失侧参数）；三臂 S6 约 1.5× 两臂成本。

## 10. 风险登记表（v3 增补）

v2 §10 全部保留，新增：

| 风险 | 概率 | 对策 |
|---|---|---|
| Geometry 头是眼动/区组读出 | 中-高 | D10 双重门前置；未通过则降级措辞，factorization 故事以 content 头为主 |
| 解纠缠正则过强伤 content（材质本身带运动特征，几何信息部分可从内容预测）| 中 | λd 从 0.1 起步消融；探针表监控 content 头朝向读出是否被压到 chance 以下的过度惩罚 |
| 审稿人问"与 EEGMirror 差异" | 高 | §2 措辞已收窄：受控因子结构 + 直接观测跨被试正样本 + 双重留出，三点 EEGMirror 都没有 |
| 多正样本内容项与 exact 项梯度冲突 | 低-中 | 两项在不同空间（trunk vs content 头），天然缓冲；监控 per-term raw loss 曲线 |

## 11. 合并裁决纪要（v2 × design.md）

| 来源 | 采纳 | 处置 |
|---|---|---|
| design.md | **竞争性不变性框架 + factorized content/geometry/semantic 多头** | 升为旗舰方法（§0/§4），已实现并通过自检 |
| design.md | EEGMirror/SAM-Net/NEED/MUSE/MB2C/CET-MAE/CI-BVCL 文献 | 已逐条核实，进 §2；新颖性措辞收窄 |
| design.md | shuffled-frame 动态性对照、结构化确定性 caption、MAE 辅助项、90/360 双层 gallery | 进 §5/§4 |
| design.md | −100~0 ms 基线校正 | **拒绝**（基线=上一试次晚期响应；v2 审查结论）|
| design.md | 0.5–45 Hz 带通 | **拒绝**（扭曲慢成分且破坏 companion 可比性；维持 0.1–100）|
| design.md | −100~700 ms 窗口 | **部分拒绝**（主窗维持 0–600 无基线；0–700 加入敏感性阶梯）|
| design.md | 随机 trial 划分警告、三协议、问卷放最后 | 与 v2 收敛，无需改动 |
| v2 | 统计推断层（视频级置换、MDD、噪声上限、FDR、40 折格）、混淆 battery、表型失谐、slurm/仓库执行层、D1–D8 | 全部保留，design.md 无对应物 |

## 12. 文献底仓

v2 §12 全部保留，追加：EEGMirror (ICCV 2025)；SAM-Net / Cross-Subject EEG-to-Video (CVPR 2026)；NEED (OpenReview)；MUSE/similarity-keeping (OpenReview KO09K3rBSr)；MB2C (ACM MM 2024)；CET-MAE/E2T-PTR (ACL 2024)；COFETT (ACL 2026, EEG-to-text 评价批判)；CI-BVCL (KBS 2026)；D²-FOSA (CVPR 2026)；NeuroCLIP (arXiv 2511.09250)；Palazzo et al. (TPAMI 2021) 及其 OpenReview 批评（"共享嵌入变好 ≠ 神经信息进入视觉表示"——factorized 探针表正是对这条批评的回应）；视觉编码器选择敏感性 (Korea Univ 2025)；Inter-subject contrastive (arXiv 2202.02901)。
