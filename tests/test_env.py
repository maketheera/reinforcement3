import numpy as np
import pytest

from game2048 import ThreeByThree2048Env


def test_reset_is_reproducible():
    first = ThreeByThree2048Env()
    second = ThreeByThree2048Env()
    observation_a, _ = first.reset(seed=12)
    observation_b, _ = second.reset(seed=12)
    assert np.array_equal(observation_a, observation_b)
    assert first.observation_space.contains(observation_a)


def test_left_merge_follows_2048_rules():
    env = ThreeByThree2048Env()
    env.reset(seed=1)
    env.board = np.array([[2, 2, 2], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    _, _, _, _, info = env.step(0)
    assert env.board[0, :2].tolist() == [4, 2]
    assert info["score"] == 4


def test_target_does_not_end_episode():
    env = ThreeByThree2048Env(max_steps=5)
    env.reset(seed=1)
    env.board = np.array([[128, 128, 0], [0, 0, 0], [0, 0, 0]], dtype=np.int32)
    _, reward, terminated, truncated, info = env.step(0)
    assert not terminated
    assert not truncated
    assert info["success"]
    assert info["max_tile"] == 256
    assert reward > 2.0


def test_invalid_move_does_not_spawn_tile():
    env = ThreeByThree2048Env()
    env.reset(seed=1)
    env.board = np.array([[2, 4, 8], [4, 8, 16], [8, 16, 32]], dtype=np.int32)
    before = env.board.copy()
    env.step(0)
    assert np.array_equal(env.board, before)


def test_truncates_at_step_limit():
    env = ThreeByThree2048Env(max_steps=1)
    env.reset(seed=2)
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated
    assert truncated
    assert info["steps"] == 1


def test_invalid_target_is_rejected():
    with pytest.raises(ValueError):
        ThreeByThree2048Env(target_tile=128)
