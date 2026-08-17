# VT-Align 蓝图：基于 ds005662 的 EEG–视频多模态对比学习

**版本**: v1（2026-08-16，文献扫描 + 数据集核查完成后的初稿）
**数据集**: "A comprehensive EEG dataset for investigating visual touch perception" (Scientific Data 13:381, 2026; OpenNeuro ds005662 v2.0.1, CC0)
**工作名**: VT-Align（Vicarious Touch Alignment；备选：TouchCLIP、VicarNet）

---

## 0. 一句话定位

利用该数据集独一无二的"**刺激少（90 个源视频）× 被试极多（80 人，全部观看完全相同的 360 个条件）**"形态，把对比学习的重心从 THINGS-EEG 式的"图像检索解码"转向三件该领域没人能做的事：

1. **被试不变的共享 EEG–视频嵌入空间**（双重对比：跨模态 EEG↔视频 + 跨被试同刺激正样本对，80 人规模是 CLISA/THINGS 系的 4–8 倍）；
2. **对齐几何即表型**（alignment-as-phenotype）：每个被试的对齐强度/子空间几何/适配器参数本身作为个体差异读出，预测替代性触觉（VT）、共情（EQ/IRI）、镜像触觉联觉（MTS）问卷分数——替代已被 N=252 重复实验证伪的 mu 抑制生物标志物；
3. **分层视频特征 × 时间分辨对齐**，裁决"社会知觉通路优先"（Lee Masson & Isik: EVC→TPJ/pSTS→体感区）vs"早期体感模拟"（Bufalari P45）两个竞争理论。

不承诺 SOTA 识别精度（Meta 缩放律论文明确：单纯加被试不提升解码，加对齐机制才有用——这恰好是我们的贡献点，而非弱点）。

---

## 1. 数据集档案（已核实，全部来自仓库实查）

### 1.1 总体
| 项 | 值 |
|---|---|
| 被试 | 80 人（54女/24男/2非二元；18–76 岁，μ=30.1）|
| EEG | 64 通道 BioSemi ActiveTwo，2048 Hz，CMS/DRL 参考，50 Hz 工频 |
| 试次 | 每人 2880 个非目标试次 + 少量目标试次（白色物体触摸，需按键）|
| 刺激 | 90 个 600 ms 无声触摸视频 × 4 朝向（原始/水平翻转/垂直翻转/双翻转）= 360 条件 × 8 重复 |
| SOA | 精确 800 ms（600 ms 视频 + 200 ms ISI），RSVP 式连续呈现，32 个序列 |
| 托管 | OpenNeuro ds005662 v2.0.1，BIDS，**CC0**，总量 ~120 GB |
| 下载 | `aws s3 sync --no-sign-request s3://openneuro.org/ds005662 .`（或 DataLad）|

### 1.2 关键文件
- `sub-XX/eeg/sub-XX_task-video_eeg.bdf`：原始连续记录，~1.3 GB/人（全部 ~110 GB，可暂缓下载）
- `derivatives/mne/sub-XX_mne_epo.fif`：**~11 GB 全套，多数工作只需要这个**。MNE 1.7 制作：平均参考、0.1–100 Hz 带通、降采样 200 Hz、epoch −100~+800 ms、基线 −100~0 ms、**无伪迹剔除/无 ICA**
- `sub-XX_task-video_events.tsv`：列含 `onsetsample, istarget, presentationnumber, stimnumber(1–384), stim(Windows 反斜杠路径), stimname, time_stimon/off, stimdur(~0.61s), rt, correct`。**朝向条件与源视频编号（1–90）从 stim 路径解析**
- `code/analysis/VTD.csv`：**90 行逐视频监督表**——arousal、threat、连续 valence（PCA 复合分）、pain(0/1)、touch_type(12类)、toucher(hand/object)、object(28类)、material(8类)、approaching(y/n)、英文动作描述（可喂文本编码器）
- `phenotype/`：EQ（15 题短版共情商数，满分30）、IRI（7 题观点采择，满分35）、VT（最富的个体差异文件：VT/noVT 分组 + 10 个探测视频逐条 Feel_touch/Sensation/Intensity/pleasant/unpleasant/painful/部位）、MTS（自报镜像触觉联觉 + 可靠性/定位自由文本）
- `participants.tsv`：直接含 VT_score、EQ_score、IRI_score、MTS 汇总列
- `code/experiment_files/stimuli/`：**全部 384 个 mp4 就在数据集里**（90×4 朝向 + 6 目标×4）
- OSF jvkqa（VTD 数据库）：另有 350 名评分者的逐视频 Neutral/Pleasant/Unpleasant/Painful 计数分布、原始长视频、验证分析 Rmd

### 1.3 工程注意事项（实查发现的坑）
- `stim` 路径是 Windows 反斜杠，需归一化；`duration` 列是名义值 0.2s，真时长看 `stimdur`
- `stimnumber` 索引 384 全集，VTD 属性按 1–90 源视频号 join（从 stimname 解析）
- SOA=800ms 且 epoch 到 +800ms：**epoch 尾部恰好触及下一刺激 onset；基线 −100~0ms 落在上一试次 onset 后 700–800ms，被上一试次晚期活动污染**——所有跨试次分析必须意识到这一点
- 600ms 视频 offset 响应（~600–750ms）在 epoch 内
- 无伪迹剔除：需自行决定伪迹策略（并把它作为消融之一）

### 1.4 已发表的分析足迹（= 我们的基线与新颖性边界）
作者组（Smit, Ramirez-Haro, Varlet, Moerel, Quek, Grootswagers；WSU MARCS）已发表两篇：
1. **Sci Data 2026**（数据集论文）：MVPA 技术验证
2. **Imaging Neuroscience 2025**（companion 论文，10.1162/IMAG.a.1017）：时间分辨 MVPA（正则化 LDA + 岭回归，留一序列 CV）。解码起始/峰值：**手朝向 ~60ms（峰 120–130ms）＞材质/物体 110–120ms＞效价起始 ~130ms（峰 300ms）＞触摸类型 165ms＞威胁/唤醒 230–260ms；疼痛 135ms 瞬态 + 240ms 起持续**。精度"modest"，靠贝叶斯统计稳健化

**全部是线性 MVPA。无深度学习、无跨被试模型、无刺激嵌入对齐、无零样本泛化。** 引用核查（Semantic Scholar 1 条=作者自引；OpenAlex 0 条）：外部无人使用。

---

## 2. 竞争格局与机会窗口

### 2.1 三个相邻社区，谁都没占住这个点
| 社区 | 现状 | 与我们的关系 |
|---|---|---|
| THINGS-EEG 对比解码（NICE→ATM→CognitionCapturer→2026 wave）| 200-way 零样本 top-1 已卷到 ~40%（被试内），但 LOSO 崩到 8–30%；全部 10 被试、图像刺激 | 方法模板来源；**最可能抢先的社区**（ds005662 已被 EEGDash 收录）|
| EEG-视频（SEED-DV/EEG2Video→DynaMind→MindCine）| 唯一公开 EEG-视频基准 SEED-DV（20 被试、需申请授权）；全在做重建 | 我们提供第二个公开 EEG-视频基准（触摸域、CC0、80 被试）|
| 替代性触觉神经科学（Smit/Keysers/Banissy/Lee Masson）| 单变量 ERP + 线性 MVPA；个体差异是作者组明示的下一步 | 科学问题来源；**作者组是个体差异角度最可能的竞争者**（但其工具箱是线性 MVPA，且 Grootswagers 离 THINGS 社区一步之遥）|

### 2.2 结论
- **对比学习角度完全空白，但窗口是时间性的**：一篇平庸的 "NICE-on-ds005662" 几个月内随时可能出现。对策：快速做出第一版基线 + 把差异化压在 80 被试才能做的事上（跨被试不变性 + 个体差异），这两点单靠移植管线做不出来。
- 唯一另一个公开"观看触摸视频 EEG"数据集：Lee Masson & Isik 2023（21 被试、75 视频、OSF 5ntcj）——规模不足以抢先，但是**现成的外部迁移测试集**（跨数据集泛化是强审稿加分项）。

---

## 3. 科学问题（论文的神经科学半边）

Q1 **观看触摸的神经编码是否因子化？** 材质×视角×效价在共享嵌入中是可分离轴还是纠缠的？4 朝向设计天然提供正交控制因子。已知风险：材质可从纯视觉统计在 <150ms 解码（Orima/Motoyoshi），效价有专用 CT/岛叶通道（Morrison 2011）——"因子化 vs 视觉混淆"本身就是可发表的问题。

Q2 **两条通路谁先承载什么？** 分层视频特征（低层运动能量/纹理 → 中层手部姿态/接触 → 高层社会-情感语义）× 时间分辨对齐权重，检验 social-perceptual-first（~90/150ms）vs 早期体感模拟（P45）。

Q3 **个体差异的表征签名**：per-subject 对齐强度/几何能否连续地预测 VT/EQ/IRI 分数？（Smit 2023：触觉→视觉跨模态分类器迁移只在自报 VT 者中成立；fMRI 表征差异性可预测共情，Imaging Neurosci 2024——EEG 时间分辨版无人做过。）

---

## 4. 模型框架

```
视频侧（冻结）                        EEG 侧（训练）
┌─────────────────────┐              ┌──────────────────────────┐
│ A. 文本对齐族:       │              │ 输入: 64ch × 180samp     │
│  SigLIP2(帧均值)     │              │ (−100~800ms @200Hz)      │
│  X-CLIP / InternVideo2│             │ per-subject EA 白化       │
│ B. 纯SSL族:          │   InfoNCE/   │ 浅层时空卷积 (NICE/ATM式) │
│  VideoMAE v2, V-JEPA2 │◄─SigLIP──► │ + 通道注意力              │
│ C. 触觉语义族:        │   masked    │ + 被试条件化:             │
│  UniTouch(ImageBind)  │             │   subject token / 1×1conv │
│  TVL 触觉形容词空间    │             │   subj-layer / SuLoRA     │
│ D. 属性/文本锚:       │             │ + MLP projector           │
│  VTD.csv 8类材质等    │             │ (0.1–4M 参数)             │
│  VLM 生成 caption     │             └──────────────────────────┘
│ E. 低层滋扰特征:      │                        │
│  光流能量/亮度/GIST   │──偏相关/回归控制────────┘
└─────────────────────┘
```

### 4.1 视频侧（全部冻结，先做编码器普查）
**第一步不是训练，是普查**：把 90 个视频过一遍 A/B/C 族编码器，算 90×90 RDM，与 VTD 行为评分 RDM（valence/arousal/threat/材质）做相关。**风险：所有视频都是同一只左手特写，通用视频模型嵌入可能坍缩**（组间方差过低）。若坍缩→转向触觉语义族（UniTouch 的 ImageBind 视觉塔、TVL 触觉形容词打分）或中层特征（光流、手部姿态 MediaPipe/HaMeR）。这一步产出论文的 "which visual representation does the brain's touch code match" 分析，本身有科学价值。

### 4.2 EEG 侧
- 起点：NICE 式浅层时空卷积 + 通道注意力（THINGS 线证明 10 被试量级大 transformer 不如浅网络；我们数据更少/被试更多，更要浅）
- 被试条件化三选一消融：subject token（CLIP-MUSED）/ 1×1 conv subject layer（Défossez）/ SuLoRA 低秩适配器。**适配器参数后续直接作为个体差异特征**（Q3）
- 预处理消融：EA（Euclidean Alignment，+4.3% 且收敛快 70%，成本近零，默认开）vs 无；伪迹策略（无/autoreject/ICA-EOG）
- EEG 基础模型（LaBraM/CBraMod/EEGPT）只做**全微调基线**，不作为主干（2025-26 基准共识：frozen probe 在 ERP 型任务接近 chance）

### 4.3 损失设计（针对 90 刺激 × 8 重复的核心方法学贡献）
批内假负样本是这个形态的根本问题（batch>90 必然含同视频"负样本"）。修复层级：
1. **label-aware masked InfoNCE / SupCon**：我们精确知道假负样本在哪（同源视频、同条件），直接掩掉或多正样本化（UniCL 形式化了 pair+label 统一目标）
2. **原型对比（ProtoNCE 式）**：单试次 EEG vs 90 个视频原型 / 360 个条件原型，天然去噪，且原型 RDM 免费供 RSA
3. **SoftCLIP 软目标**：对齐目标 = 视频编码器相似度行向量而非 one-hot，注入分级相似结构
4. **SigLIP sigmoid 逐对损失**：小 batch 友好，且允许逐对设置正/掩标签——消融 sigmoid vs masked-softmax vs SupCon
5. **双重对比（差异化核心）**：L = L_cross-modal(EEG↔video) + λ₁·L_cross-subject(CLISA 式同刺激异被试正样本，80 人规模史无前例) + λ₂·L_RNC(Rank-N-Contrast 连续效价轴) + λ₃·L_text(caption/属性锚，NICE→NICE-LLM 证明 +4pts)

### 4.4 第三模态（文本/属性）
- VTD.csv 英文动作描述 + VLM 对 90 视频生成结构化 caption（toucher/object/material/force/hedonic tone）
- 属性锚：材质 8 类、touch_type 12 类文本嵌入；连续 valence/arousal/threat 用 RNC
- CEBRA 式属性条件采样作为另一实现路径（正样本按共享属性/相近效价采样）

---

## 5. 评测协议（先定协议后跑实验，防 p-hacking 也防审稿人）

### 5.1 数据划分（最重要的设计决定）
- **零样本 = 源视频不相交（base-video-disjoint）**：held-out 视频的全部 4 朝向 × 8 重复 × 全体被试整体进测试侧；按材质/touch_type 分层；**朝向级或重复级划分 = 泄漏**（同一素材翻转而已），等价于 THINGS-EEG2 的 concept-disjoint 原则
- 90 个源视频 → 测试集 ~9–18 个 → 零样本仅 9–18-way，统计脆弱 → **重复随机划分（≥10 折视频层面 CV）+ 置换检验估计 chance**，不用单一固定划分
- 三种被试机制分开报：被试内 / LOSO（79 人训练，比文献的 9 人 LOSO 强一个量级）/ LOSO+少量校准（adapter/SRM 投影拟合）
- 双向留出（video-disjoint × subject-disjoint）为最强声明

### 5.2 指标
- 零样本检索：top-1/top-5，gallery 大小 2/10/18/90 梯度；单试次与 2–4 试次伪试次平均**双报**（伪试次文献：平均 ~4 试次最优，重平均反而伤被试间方差）
- 属性零样本迁移：材质 8-way、视角 2-way（**必须跨朝向泛化**，见 6.3）、效价回归 r
- RSA 轨道：时间分辨 EEG RDM（360 条件）vs 视频编码器 RDM / 属性 RDM / 低层特征 RDM，偏相关剥离低层
- 嵌入空间探针：被试身份分类精度（验证不变性——越低越好）+ 属性线性可读性
- 个体差异：per-subject 对齐分数 → VT/EQ/IRI 相关。**n=80 在 r≈0.31 才有 80% 功效，预注册为次要/探索性终点；MTS 真阳性预计 1–2 人，只做个案描述不做推断**

### 5.3 基线阶梯（缺一审稿必问）
1. companion 论文的 LDA/岭回归（同属性、同 CV 逻辑）
2. 线性被试不变基线：CorrCA/ISC 空间滤波、SRM（80 被试 × 360 共享条件是 SRM 理想领地；新被试只需拟合投影）
3. EA + 线性
4. NICE、ATM 移植（公开代码）
5. LaBraM/CBraMod/EEGPT 全微调
6. 我们的 VT-Align 完整模型 + 逐组件消融

---

## 6. 混淆控制清单（2024–26 批判文献后的必备品）

1. **区组/时间自相关**（Li 2021 TPAMI"block design"、watermelon 2024）：实查 events.tsv 确认条件在序列内充分交错；trial-index 解码器对照（用试次序号"解码"标签，要求 chance）；时间置换零分布
2. **低层视觉混淆**（NeuroImage 2024 证明 THINGS 语义解码部分是低层统计）：光流能量/亮度/对比度/手位置回归器；偏相关 RSA；SATED 教训——运动能量与唤醒相关，必须先剥离再谈"情感对齐"
3. **眼动**：触摸视频有系统性注视吸引子（接触点、运动手），且随朝向/视角变——恰是我们的标签。EOG-only 解码器基线（EEG 对齐必须显著超过它）；无 EOG 通道 → 用 Fp1/Fp2/F7/F8 前额通道近似 + ICA 眼动成分分析
4. **视角≠翻转**：数据集的 self/other 由垂直翻转操作化，与最强最早的朝向信号（60ms 起）混淆。视角声明必须做**跨朝向泛化**（在一种翻转上训练、另一种上测试）
5. **epoch 污染**：基线窗含上一试次晚期响应；800ms SOA 下相邻试次重叠——考虑加做 0–600ms 窗口敏感性分析
6. **重建克制**：90 个同域视频做生成式"重建"正中 Shirakawa 2024 spurious-reconstruction 批评靶心——**只做检索/辨识，不做重建**（最多附录演示并明确标注局限）

---

## 7. 路线图（相位 + go/no-go）

### Phase 0 — 数据落地与体检（1–2 周）
- 只下 derivatives (~11GB) + events + phenotype + stimuli + VTD.csv
- 复现 companion 论文 2–3 条解码曲线（朝向/材质/效价）→ 验证数据管道正确
- events.tsv 随机化审计（混淆清单 #1）
- 视频编码器普查（4.1）：90×90 RDM vs 行为 RDM
- **Go/no-go**：至少一族视频编码器的 RDM 与行为评分显著相关（否则先解决视频表征再谈对齐）

### Phase 1 — 最小可行对齐（2–4 周）
- NICE 移植 + 正确划分协议 + 评测 harness（5.1/5.2 全套先写好）
- 被试内零样本检索显著高于 chance → 核心可行性确立
- 同时跑混淆对照 battery（一次性代码，终身受用）
- **Go/no-go**：被试内 18-way top-1 显著 + EOG-only 基线被超过

### Phase 2 — 方法创新（4–8 周）
- 损失阶梯消融（4.3 的 1→5）；被试条件化三选一；EA/伪迹消融
- 双重对比引入：LOSO 曲线随训练被试数（10/20/40/79）的缩放——**无论结果方向都是主图**（正：对齐机制把队列规模变成跨被试性能，填补 Meta 缩放律指出的空白；负：EEG 版缩放律，同样可发表）

### Phase 3 — 科学分析（与 Phase 2 并行推进）
- 分层视频特征 × 时间分辨对齐（Q2）；因子化几何分析（Q1）
- 个体差异：per-subject 对齐分数/适配器参数 → VT/EQ/IRI（Q3，预注册式声明为探索性）

### Phase 4 — 外部验证与成文
- Lee Masson & Isik 数据集（OSF 5ntcj）跨数据集迁移
- （可选）Smit 2023 felt-touch 数据集：把"体感模拟"操作化为嵌入空间邻近度
- （可选）FACED（123 被试情感视频）作为辅助预训练语料

### 投稿目标（2026-08 视角）
- 主选：**ICLR 2027**（摘要 ddl 通常 9 月中下旬——如赶不上完整故事，可只带 Phase 0–2 核心结果）或 **NeurIPS 2027**；AAAI-28 备选
- 科学半边独立成文可投 Imaging Neuroscience / J Neurosci（与 ML 论文互引不冲突）
- 若竞速压力大：Phase 1 结果先挂 arXiv 占位

---

## 8. 风险登记表

| 风险 | 概率 | 对策 |
|---|---|---|
| 被 "NICE-on-ds005662" 抢先 | 中 | 速度 + 差异化压在 80 被试专属贡献；Phase 1 后即挂 arXiv |
| 作者组先发个体差异分析 | 中 | 我们的个体差异走表征几何路线（他们是线性 MVPA 相关分析）；引用并定位为互补 |
| 视频嵌入坍缩（同手同景）| 中 | Phase 0 普查前置；备胎=触觉语义族/中层特征/微调视频头 |
| 单试次 SNR 不足以支撑 InfoNCE | 中 | 原型对比 + 伪试次训练策略 + companion 论文已证明信号存在（只是弱）|
| 零样本 18-way 统计脆弱 | 高（必然）| 重复划分 CV + 置换 chance + 属性级零样本作为共同主指标 |
| 跨被试缩放不涨（Meta 缩放律）| 中 | 预注册双向解读：正=方法贡献，负=EEG 缩放律实证，都可发表 |
| n=80 个体差异功效不足 | 高 | 声明探索性；用试次级混合模型增功效；效应量置信区间而非二值检验 |
| 审稿人拿 THINGS 200-way 数字对比 | 中 | 明确框定：不同刺激词表规模不可比；报 gallery-size 梯度曲线 |

---

## 9. 关键文献底仓（按用途）

**方法模板**：NICE (ICLR'24, github: eeyhsong/NICE-EEG)；ATM (NeurIPS'24, dongyangli-del/EEG_Image_decode)；CognitionCapturer (AAAI'25)；NICE-LLM (TNNLS'25，文本+4pts)；EEG2Video/SEED-DV (NeurIPS'24)；CLIP-MUSED (ICLR'24, subject tokens)；Défossez 2023 (Nat MI, subject layer)；SuLoRA (arXiv:2510.08059)；CLISA (TAC'22)；CEBRA (Nature'23)；SupCon (NeurIPS'20)；UniCL (CVPR'22)；ProtoNCE (ICLR'21)；SoftCLIP (AAAI'24)；SigLIP (ICCV'23)；Rank-N-Contrast (NeurIPS'23)
**视频/触觉编码器**：VideoMAE v2；V-JEPA 2 (facebookresearch/vjepa2，attentive-probe 协议)；InternVideo2 (f4 变体适配 600ms)；X-CLIP；SigLIP2；UniTouch (CVPR'24)；TVL (ICML'24)；Touch-and-Go (NeurIPS'22)
**被试不变/对齐**：EA (He&Wu'20 + Junqueira'24 系统评估)；SRM (NeurIPS'15, brainiak)；CorrCA/ISC (Parra lab)；SREA (arXiv:2602.01728)；跨被试综述 (arXiv:2604.27033)
**尺度定标**：Meta scaling laws (arXiv:2501.15322)；THINGS-EEG2 (NeuroImage'22)；LOSO 数字汇总 (MindAlign arXiv:2605.24523)
**混淆批判（审稿人手册）**：Li 2021 TPAMI block-design；Ahmed CVPR'21；watermelon (arXiv:2405.17024)；低层统计混淆 (NeuroImage'24)；Shirakawa spurious reconstruction (arXiv:2405.10078)；伪试次优化 (bioRxiv 2023.10.04.560678)；眼动混淆 (eNeuro'18)
**神经科学**：companion 论文 (Imaging Neurosci'25, PMC12624366)；Smit 2023 felt/seen (NeuroImage)；Smit 2025 VT 普查 (Sci Rep, 84% 普遍性)；Lee Masson & Isik 2023 (J Neurosci + OSF 5ntcj)；VTD (Behav Res Methods'25)；Keysers 2010 (NRN)；Blakemore 2005 (Brain)；mu 抑制零结果 (Neuropsychologia'19, N=252)；vicarious body maps (Nature'25)
**EEG 基础模型（仅基线）**：LaBraM (ICLR'24)；EEGPT (NeurIPS'24)；CBraMod (ICLR'25)；FM 怀疑论基准 (EEG-FM-Bench arXiv:2508.17742 等)
