# 3x3 2048 强化学习

这是一个可独立上传 GitHub、下载到服务器运行的 3×3 2048 强化学习项目。环境遵循 2048 的移动、合并和随机生成规则，智能体使用带合法动作屏蔽的 Double DQN。

## 目标定义

- 成功：一局中棋盘最大块达到或超过 256。
- 一局不会在达到 256 时提前结束，继续玩到棋盘无合法移动或达到步数上限。
- 主要评估指标：成功率；次级指标：成功局的终局棋盘数字总和。
- 训练奖励包含合并得分、最大块增长、空格数量变化、有效移动奖励、无效动作惩罚和首次达到 256 的奖励。
- 训练和评估都会屏蔽无效动作；Double DQN 的下一状态目标 Q 值也只从合法动作中计算。

> 注意：在标准 2048 规则下，棋盘总和只会因新生成的 2/4 增长；合并不改变总和。因此终局总和主要衡量存活并完成了多少次有效移动，而非合并分数。

## 新建 GPU 环境（Mamba + pip）

之前的 Conda 方案解析到了 CPU 构建。现在使用 Mamba 安装 Python、NumPy、Gymnasium、pytest；再用 pip 从南京大学 PyTorch CUDA wheel 镜像安装固定的 `torch 2.6.0+cu124`，并从清华 PyPI 镜像取通用依赖。此方案不会从 Conda 安装 PyTorch。

项目已配置独立环境名 `reinforcement3-pipcuda`，不要覆盖旧环境。先在项目根目录创建基础环境并激活：

```bash
source ~/miniforge3/etc/profile.d/conda.sh
mamba env create -f environment.yml
conda activate reinforcement3-pipcuda
```

然后安装 CUDA PyTorch：

```bash
python -m pip install -r requirements-gpu.txt --timeout 600 --retries 8
```

配置的一键脚本 `bash scripts/setup_mamba_gpu.sh` 会在目标环境已激活时执行 pip 安装并验证 CUDA；若没有激活目标环境且 shell 找不到 mamba，会给出提示。

确认不是 CPU 版：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.ones(3, device='cuda').sum().item())"
```

应看到 `2.6.0+cu124`、`12.4`、`True` 和结果 `3.0`。pip 安装前可用 `--dry-run` 验证候选包：

```bash
python -m pip install --dry-run -r requirements-gpu.txt
```

如果 NJU 镜像访问失败，先不要改成普通 PyPI 镜像并安装不带 `+cu124` 的 torch；把报错贴出来再确认可用的 CUDA wheel 源。不要让 pip 和 Conda 同时管理 PyTorch。

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
