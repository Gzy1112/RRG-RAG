# LLaMA-Factory (modified)

Vendored copy of [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) `0.9.1.dev0` (Apache-2.0, see `LICENSE`), as extended by MMOA-RAG for multi-agent PPO. It is used for every training stage of RRG-RAG.

You do not need to install it: the launchers in `../scripts/` run it from `src/`.

## Additions for RRG-RAG

| File | Purpose |
|---|---|
| `src/train_restore.py` | Role-Specialized Sequential Optimization, Synthesizer round (per-agent PPO, `VIDORAG_REWARD_METHOD`) |
| `src/train_restore_7b.py` | Role-Specialized Sequential Optimization, Seeker round. The frozen Synthesizer used by the end-to-end reward is placed on `VIDORAG_E2E_SYNTH_DEVICE`. |
| `src/train_restore_grpo.py` | GRPO-Based Inspector Controller Optimization (`VIDORAG_MAPPO_TRAINER=inspector_grpo_v2`) |
| `src/llamafactory/train/ppo/workflow_restore*.py` | Trainer and reward selection, gold-answer lookup, per-agent PPO loop |
| `src/llamafactory/train/ppo/trainer_vidorag_restore*.py`, `trainer_vidorag.py` | Qwen2.5-VL multi-image PPO trainer for the agents |
| `src/llamafactory/train/ppo/vidorag_rewards*.py`, `normalize_*.py`, `inspector_filter.py` | Role-specific rewards (evidence, answer F1, Synthesizer reward) |
| `src/llamafactory/train/ppo/trainer_vidorag_controller_grpo.py`, `trainer_vidorag_inspector_grpo_v2.py` | GRPO over cached counterfactual `STOP` / `RETRIEVE-MORE` branch outcomes (Inspector; `RETRIEVE-MORE` is `DECK` in the code) |
| `examples/train_lora/qwen2_5_vl_7b_*.yaml` | SFT / Role-Specialized Sequential Optimization / GRPO-Based Inspector Controller Optimization configs |
| `data/dataset_info.json` | SFT / PPO datasets (built by `../data_prep/`) |

`DISABLE_VERSION_CHECK=1` is required: the tested environment (`../requirements.txt`) is newer than the upstream version pins in `requirements.txt`.
