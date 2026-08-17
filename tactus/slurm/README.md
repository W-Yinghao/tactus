# TACTUS on slurm

一切计算通过作业提交，登录节点只做提交与检视。数据根目录 `/projects/EEG-foundation-model`。

## 启动顺序

```bash
bash slurm/cluster_probe.sh | tee slurm/cluster_report.txt
```

把报告里的 account / QOS / 环境激活方式填进 `slurm/cluster.conf`（`ENV_SETUP` 一行填服务器现有的激活命令，例如 `source /opt/conda/etc/profile.d/conda.sh && conda activate eegfm`）。

然后在**计算节点**上验环境——登录节点通常看不到 GPU，在那里跑 `env_probe.py` 会误报无卡：

```bash
srun -p L40S --gres=gpu:1 -t 00:10:00 --pty python env_probe.py --skip-disk
```

确认无误后看一眼作业图，再真提交：

```bash
python slurm/submit.py --chain all --dry-run
python slurm/submit.py --chain all
```

## 作业图

```
download(array 1-81) → trials → preprocess(array 1-80) ┬→ mvpa(array 1-80) ┐
                            download → embed(gpu) ─────┴→ train(array/fold)─┴→ eval
```

依赖用 `--dependency=afterok` 串联，所以一次提交即可无人值守跑完整条链。生成的 sbatch 脚本落在 `slurm/generated/`，提交前可直接阅读修改。

## 分区选择策略（重要）

TACTUS 的 EEG 编码器是 **0.1–4M 参数**，输入 `(64, 120)`，batch 256 只有约 8 MB。它是**数据加载瓶颈而非算力瓶颈**——排队等 H100 的时间远超省下的计算时间。因此：

| 阶段 | 分区 | 理由 |
|---|---|---|
| download | `CPU` | 网络瓶颈，80 路并行 s3 sync |
| preprocess | `cpu-high` | 读 1.3 GB BDF + 2048 Hz 滤波 + ICA，吃内存（64 G/任务） |
| embed | `L40S / A40 / V100` | 360 个 600 ms 片段过冻结编码器，几分钟 |
| **train** | **`L40S / A40 / V100`** | **主力扫描放小卡，队列短、完全够用** |
| mvpa / eval | `cpu-high` | 纯 sklearn 与统计聚合 |
| fm（可选） | `H100 / A100` | 只有 LaBraM/CBraMod/EEGPT 全微调和十亿级视频编码器值得占大卡 |

`submit.py` 会用 `sinfo` 探测候选分区的空闲情况，在 `GPU_SMALL_PARTITIONS` 列表里挑最闲的一个，并打印它为什么改了选择。

## 抢占与重跑

所有阶段幂等可续跑（已存在的输出跳过），因此 `cluster.conf` 里默认开 `REQUEUE=1`。作业被抢占或超时后 slurm 自动重排，脚本从断点继续，不会写坏中间产物。

数组作业都带节流（`%N`），默认值偏保守。队列空的时候把 `TRAIN_ARRAY_THROTTLE` 和 `PREPROCESS_ARRAY_THROTTLE` 调高即可——40 折的双重留出格在 6 并发下约 10 轮跑完，调到 20 并发则是 2 轮。

## 常用

```bash
python slurm/submit.py --status                                  # 本项目队列
python slurm/submit.py --stage train --config configs/nice_protonce.yaml   # 只重跑训练
python slurm/submit.py --stage train --regime double_disjoint    # 40 折主声明格
tail -f /projects/EEG-foundation-model/tactus_work/logs/train/*.out
sacct -X --format=JobID,JobName%22,Partition,State,Elapsed,MaxRSS -S today
```

换对比学习算法只改 config 里的 `loss:` 一个键，其余全部不动：

```bash
python slurm/submit.py --stage train --config configs/my_new_loss.yaml
```
