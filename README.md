# Pacman Agent with Double DQN + Prioritized Experience Replay

*PyTorch · Gymnasium · TensorBoard · OpenCV*

A reinforcement learning agent trained to play Ms. Pacman using Double DQN with Prioritized Experience Replay, built from scratch without high-level RL libraries.

---

## Highlights

- Implemented Double DQN to decouple action selection from value estimation, reducing overestimation bias inherent in vanilla DQN
- Built Prioritized Experience Replay from scratch using a SumTree data structure with importance sampling weights to correct the introduced sampling bias
- Applied frame stacking (4 consecutive grayscale frames, 84×84) to give the agent temporal context for velocity and direction
- Added reward shaping (+0.01 per step) to encourage survival alongside score maximization
- Tracked training with TensorBoard**: episode reward, loss, epsilon decay, and 100-episode rolling average

---

## Architecture

```
Raw Frame (RGB) =====> Grayscale → Resize (84×84) → Frame Stack (×4) =====>CNN: Conv(32, 8×8, s4) → Conv(64, 4×4, s2) → Conv(64, 3×3, s1) =====> FC: 3136 → 512 → n_actions =====> Double DQN + PER update
```

**Policy network** selects actions; **target network** evaluates them — updated every 10,000 steps via hard copy.

---

## Key Implementation Details

| Component | Detail |
|---|---|
| Replay buffer | SumTree-based PER, capacity 100k |
| PER alpha | 0.6 (priority exponent) |
| PER beta | 0.4 → 1.0 (IS correction, annealed) |
| Optimizer | Adam, lr = 5e-5 |
| Epsilon decay | 1.0 → 0.01 over 220k steps |
| Batch size | 32 |
| Gamma | 0.99 |
| Target update | Every 10,000 steps |
| Grad clipping | max norm = 10 |

---

## Getting Started

**Requirements:** Python 3.7+, CUDA optional

```bash
git clone https://github.com/sonplusplus/pacman-double-dqn.git
cd pacman-double-dqn
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Train:**
```bash
python main.py --mode train
# Custom hyperparameters:
python main.py --mode train --args --lr 0.0001 --train-steps 500000 --eps-decay 100000
```

**Test a saved checkpoint:**
```bash
python main.py --mode test --args --model-path checkpoints/<timestamp>/model_best.pt --episodes 10 --render
# Record as GIF:
python main.py --mode test --args --model-path checkpoints/<timestamp>/model_best.pt --record
```

**Monitor training:**
```bash
tensorboard --logdir logs
# Open http://localhost:6006
```

---

## Key Training Arguments

| Argument | Default | Description |
|---|---|---|
| `--lr` | 5e-5 | Learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--eps-start` | 1.0 | Initial epsilon |
| `--eps-final` | 0.01 | Final epsilon |
| `--eps-decay` | 220000 | Decay steps |
| `--train-steps` | 1,000,000 | Total training steps |
| `--buffer-size` | 100000 | Replay buffer capacity |
| `--target-update` | 10000 | Target network update interval |


---

## Contact

Vu Ngoc Son · [vungocson189204@gmail.com](mailto:vungocson189204@gmail.com) · [github.com/sonplusplus](https://github.com/sonplusplus)
