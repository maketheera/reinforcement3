# 3x3 2048 强化学习

这是一个可独立上传 GitHub、下载到服务器运行的 3×3 2048 强化学习项目。环境遵循 2048 的移动、合并和随机生成规则，智能体使用带合法动作屏蔽的 Double DQN。

## 目标定义

- 成功：一局中棋盘最大块达到或超过 256。
- 一局不会在达到 256 时提前结束，继续玩到棋盘无合法移动或达到步数上限。
- 主要评估指标：成功率；次级指标：成功局的终局棋盘数字总和。
- 训练奖励包含合并得分、最大块增长、空格数量变化、有效移动奖励、无效动作惩罚和首次达到 256 的奖励。
- 训练和评估都会屏蔽无效动作；Double DQN 的下一状态目标 Q 值也只从合法动作中计算。

> 注意：在标准 2048 规则下，棋盘总和只会因新生成的 2/4 增长；合并不改变总和。因此终局总和主要衡量存活并完成了多少次有效移动，而非合并分数。

## 新建 GPU 环境（推荐）

不要继续在原来的 `qhxx_env` 中混装 Conda 与 pip 的 PyTorch。项目使用独立环境 `reinforcement3-gpu`：Mamba 负责 Python、NumPy、Gymnasium 和 pytest，pip 从 PyTorch 官方 CUDA 12.4 wheel 源安装 PyTorch 2.6.0。服务器的 NVIDIA Driver 570.133.07 支持 CUDA 12.4 runtime。

从项目根目录执行：

```bash
mamba env create -f environment.yml
mamba run -n reinforcement3-gpu python -m pip install --upgrade pip
mamba run -n reinforcement3-gpu python -m pip install -r requirements-gpu.txt --timeout 600 --retries 10
mamba activate reinforcement3-gpu
```

也可执行一键脚本：

```bash
bash scripts/setup_mamba_gpu.sh
```

检查安装结果：

```bash
python -c "import torch; print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('available=', torch.cuda.is_available()); print('count=', torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

必须看到 `torch_cuda=12.4`、`available=True` 和两张 RTX 3080。若不满足，不要训练。测试环境：

```bash
python -m pytest -q
```

旧环境不需要删除；新环境成功后可停止使用 `qhxx_env`。
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
