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

"""7B dual-GPU TFR entry point.

This is a variant of train_restore.py that uses workflow_restore_7b.py
to support loading the frozen Synthesizer on a separate GPU (cuda:1)
for 7B TFR where Round B needs 2x 7B models.

Environment variables:
    VIDORAG_E2E_SYNTH_DEVICE: Device for frozen Synthesizer (default: "cuda:0")
                              For 7B TFR, set to "cuda:1" with CUDA_VISIBLE_DEVICES=0,1
    VIDORAG_SKIP_DISTRIBUTED_CHECK: Set to "1" to bypass distributed training check
                                    (required for single-process dual-GPU mode)

Usage:
    CUDA_VISIBLE_DEVICES=0,1 VIDORAG_E2E_SYNTH_DEVICE=cuda:1 VIDORAG_SKIP_DISTRIBUTED_CHECK=1 \
        python src/train_restore_7b.py <config.yaml>
"""

from llamafactory.train.callbacks import LogCallback
from llamafactory.hparams import get_train_args


def main():
    callbacks = [LogCallback()]
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(None)

    if finetuning_args.stage == "ppo":
        from llamafactory.train.ppo.workflow_restore_7b import run_ppo as _run_ppo_restore_7b
        _run_ppo_restore_7b(
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            finetuning_args=finetuning_args,
            generating_args=generating_args,
            callbacks=callbacks,
        )
    else:
        raise ValueError(
            "train_restore_7b.py only supports PPO stage. "
            "Use train.py for other stages (sft, pt, rm, dpo, kto)."
        )


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
