# 3x3 2048 强化学习

这是一个可独立上传 GitHub、下载到服务器运行的 3×3 2048 强化学习项目。环境遵循 2048 的移动、合并和随机生成规则，智能体使用带合法动作屏蔽的 Double DQN。

## 目标定义

- 成功：一局中棋盘最大块达到或超过 256。
- 一局不会在达到 256 时提前结束，继续玩到棋盘无合法移动或达到步数上限。
- 主要评估指标：成功率；次级指标：成功局的终局棋盘数字总和。
- 训练奖励包含合并得分、最大块增长、空格数量变化、有效移动奖励、无效动作惩罚和首次达到 256 的奖励。
- 训练和评估都会屏蔽无效动作；Double DQN 的下一状态目标 Q 值也只从合法动作中计算。

> 注意：在标准 2048 规则下，棋盘总和只会因新生成的 2/4 增长；合并不改变总和。因此终局总和主要衡量存活并完成了多少次有效移动，而非合并分数。

## GPU 环境（Linux x86_64）

之前的 2.6.0 Conda 配置有误：PyTorch 从 2.6 起不再发布官方 Conda 二进制包，求解器选中了 conda-forge 的 CPU 构建。安装 `pytorch-cuda` 元包不能改变 CPU 版 PyTorch。

现在固定官方渠道的 **PyTorch 2.5.1 CUDA 12.4 构建**，新环境名为 `reinforcement3-cu124`。构建约束禁止回退 CPU；找不到匹配包时会报错，不应移除约束。

先预览事务，不下载、不创建环境：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda deactivate
mamba env create --strict-channel-priority --dry-run -f environment.yml
```

确认 `pytorch` 的版本为 `2.5.1`，构建名包含 `cuda12.4`，渠道为 `pytorch`。不能是 `cpu_mkl`。确认后再安装：

```bash
mamba env create --strict-channel-priority -f environment.yml
conda activate reinforcement3-cu124
bash scripts/setup_mamba_gpu.sh
```

脚本会检查 CUDA 并执行真实 GPU 张量运算。已下载的包可能复用缓存，请保留缓存和原环境；不要覆盖当前激活环境。不要再通过 pip 安装第二套 PyTorch。

也可手动检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.ones(3, device='cuda').sum().item())"
```

预期为 `2.5.1`、`12.4`、`True`、`3.0`。本机没有 Mamba/Linux GPU，配置未实际求解或安装；以服务器 dry-run 及 GPU 运算验证结果为准。

## 验证环境

```bash
python -m pytest -q
python scripts/play_random.py
```

## 小规模训练检查

```bash
python scripts/train.py --episodes 100 --max-steps 100 --device cuda
```

观察输出中的 `Using GPU`、`success_rate_100` 和 `loss`。确认无误后再开始长时间训练。

## 训练

```bash
python scripts/train.py --episodes 50000 --max-steps 500 --seed 42 --device cuda
```

最佳模型保存在 `checkpoints/best.pt`。模型按是否成功优先、然后按终局棋盘总和更新。建议使用至少 3 个随机种子分别训练或评估。

## 评估

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 100 --seed 10000 --device cuda
```

重点看 `success_rate` 和 `avg_success_board_sum`。用不同 `--seed` 重复评估，可减少单组随机种子的偶然性。

## GitHub 上传建议

源代码和测试应上传；模型 checkpoint 可单独保存。若确认模型文件较小，也可以：

```bash
git add src scripts tests README.md requirements.txt requirements-gpu.txt environment.yml
 git add -f checkpoints/best.pt
git commit -m "Improve GPU training and legal action masking"
git push
```

如果不想把模型放进普通 Git 历史，删除 `git add -f checkpoints/best.pt`，只上传代码。

## 项目结构

```text
reinforcement3/
  src/game2048/env.py       # 3x3 环境、奖励和合法动作
  src/game2048/dqn.py       # Double DQN、动作屏蔽和经验回放
  scripts/train.py          # GPU 训练入口
  scripts/evaluate.py       # 多局评估
  scripts/play_random.py    # 随机策略环境检查
  tests/test_env.py         # 环境规则测试
```
