# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO variant entry point for MAPPO independent-adapter training.

Identical to src/train_restore.py except it routes PPO stage to
workflow_restore_grpo, which imports ViDoRAGVLMPPOTrainer from
trainer_vidorag_restore_grpo (the GRPO-enabled trainer). The original
train_restore.py and workflow_restore.py are left unchanged.

Usage:
    python src/train_restore_grpo.py <config.yaml>

Set MAPPO_USE_GRPO=1 in the environment to activate the critic-free GRPO
policy-gradient path; otherwise the original PPO path runs.
"""

from llamafactory.train.callbacks import LogCallback
from llamafactory.hparams import get_train_args


def main():
    callbacks = [LogCallback()]
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(None)

    if finetuning_args.stage == "ppo":
        from llamafactory.train.ppo.workflow_restore_grpo import run_ppo as _run_ppo_restore_grpo
        _run_ppo_restore_grpo(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            generating_args=generating_args,
            callbacks=callbacks,
        )
    else:
        raise ValueError(
            "train_restore_grpo.py only supports PPO stage. "
            "Use train.py for other stages (sft, pt, rm, dpo, kto)."
        )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
