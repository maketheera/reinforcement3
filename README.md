# 3x3 2048 强化学习

这是一个可独立上传 GitHub、下载到服务器运行的 3×3 2048 强化学习项目。环境遵循 2048 的移动、合并和随机生成规则，智能体使用带合法动作屏蔽的 Double DQN。

## 目标定义

- 成功：一局中棋盘最大块达到或超过 256。
- 一局不会在达到 256 时提前结束，继续玩到棋盘无合法移动或达到步数上限。
- 主要评估指标：成功率；次级指标：成功局的终局棋盘数字总和。
- 训练奖励包含合并得分、最大块增长、空格数量变化、有效移动奖励、无效动作惩罚和首次达到 256 的奖励。
- 训练和评估都会屏蔽无效动作；Double DQN 的下一状态目标 Q 值也只从合法动作中计算。

> 注意：在标准 2048 规则下，棋盘总和只会因新生成的 2/4 增长；合并不改变总和。因此终局总和主要衡量存活并完成了多少次有效移动，而非合并分数。

## 环境安装

建议使用 Python 3.10。`requirements.txt` 将 Gymnasium 固定为 `0.29.1`，因为 Gymnasium 1.4 在部分 Python 3.10 环境中会触发 `typing` 泛型兼容错误。

```bash
cd reinforcement3
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## GPU 检查与 PyTorch

先检查 NVIDIA 驱动：

```bash
nvidia-smi
```

你的服务器显示 NVIDIA Driver `570.133.07`、CUDA `12.8`。安装依赖后检查 PyTorch 是否能调用 GPU：

```bash
python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('count=', torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

必须看到 `available=True`。如果仍然是 `False`，不要直接开始训练；当前 PyTorch 可能是 CUDA 13 构建，与服务器驱动不匹配。请按 PyTorch 官网选择与 CUDA 12.x/驱动兼容的安装命令，或安装服务器管理员提供的 CUDA 12.x 版本 PyTorch。

训练时使用：

```bash
python scripts/train.py --episodes 50000 --max-steps 500 --seed 42 --device cuda
```

脚本会打印 GPU 名称；如果显式指定 `--device cuda` 但 GPU 不可用，程序会立即报错，不会静默退回 CPU。

如果暂时只能使用 CPU：

```bash
python scripts/train.py --episodes 50000 --max-steps 500 --seed 42 --device cpu
```

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
git add src scripts tests README.md requirements.txt
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
