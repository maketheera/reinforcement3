# 3x3 2048 强化学习

这是一个可独立上传 GitHub、下载到服务器运行的 3×3 2048 强化学习项目。环境遵循 2048 的移动、合并和随机生成规则，智能体使用 Double DQN。

## 目标定义

- 成功：一局中棋盘最大块达到或超过 256。
- 一局不会在达到 256 时提前结束，继续玩到棋盘无合法移动或达到步数上限。
- 主要评估指标：成功率；次级指标：成功局的终局棋盘数字总和。另报告总体平均棋盘和、最大块与步数。
- 训练奖励鼓励合并得分和成功，同时对有效移动给小额奖励、无效移动给小额惩罚。

> 注意：在标准 2048 规则下，棋盘总和只会因新生成的 2/4 增长；合并不改变总和。因此终局总和主要衡量存活并完成了多少次有效移动，而非合并分数。报告成功率与成功局总和，避免单看总和掩盖未达到 256 的回合。

## 环境安装

建议使用 Python 3.10。`requirements.txt` 将 Gymnasium 固定为 `0.29.1`，因为 Gymnasium 1.4 在部分 Python 3.10 环境中会触发 `typing` 泛型兼容错误。

```bash
cd reinforcement3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

如果当前环境已经安装过 Gymnasium 1.4，请先执行：

```bash
pip uninstall -y gymnasium
pip install "gymnasium==0.29.1"
```

如服务器需要 GPU，请先按 PyTorch 官方安装说明安装与 CUDA 匹配的 PyTorch，再安装其余依赖。CPU 可正常训练，但速度较慢。

## 验证环境

```bash
python -m pytest -q
python scripts/play_random.py
```

## 训练

```bash
python scripts/train.py --episodes 20000 --max-steps 500 --seed 42 --device auto
```

先小规模检查运行：

```bash
python scripts/train.py --episodes 10 --max-steps 100 --device cpu
```

最佳模型保存在 `checkpoints/best.pt`。模型按是否成功优先、然后按终局棋盘总和更新。训练不是性能保证；请用多个随机种子评估。

## 评估

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 100 --seed 10000 --device auto
```

逐步输出棋盘：

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 1 --render
```

重点看 `success_rate` 和 `avg_success_board_sum`。用不同 `--seed` 重复评估，可减少单组随机种子的偶然性。

## GitHub 上传建议

在 `reinforcement` 父目录下：

```bash
git add reinforcement3
git commit -m "Add 3x3 2048 Double DQN project"
```

如果父目录还没有 Git 仓库，再先执行 `git init`。不要上传 `.venv`、训练 checkpoint 或运行日志；`.gitignore` 已排除这些内容。

## 项目结构

```text
reinforcement3/
  src/game2048/env.py       # Gymnasium 3x3 环境
  src/game2048/dqn.py       # Double DQN 与经验回放
  scripts/train.py          # 训练入口
  scripts/evaluate.py       # 多局评估
  scripts/play_random.py    # 随机策略环境检查
  tests/test_env.py         # 环境规则测试
```
