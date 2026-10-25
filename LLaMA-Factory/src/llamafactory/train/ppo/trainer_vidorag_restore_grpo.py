# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# GRPO variant of trainer_vidorag_restore.py. Identical to the base trainer
# except that the PPO value head is bypassed: advantages are computed
# group-relative (GRPO) instead of via a learned critic. Enable by setting
# MAPPO_USE_GRPO=1; the original PPO path is retained as the default.
# The base file trainer_vidorag_restore.py is left unchanged.
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

import math
import os
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer import DEFAULT_CALLBACKS
from transformers.trainer_callback import CallbackHandler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits
from trl.models.utils import unwrap_model_for_generation
from typing_extensions import override
from PIL import Image

from ...extras.logging import get_logger
from ...extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .ppo_utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm

import re
import time
import json
import random
import requests
from collections import Counter
from transformers import LogitsProcessorList, LogitsProcessor


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import (
        DataCollatorWithPadding,
        PreTrainedTokenizer,
        ProcessorMixin,
        Seq2SeqTrainingArguments,
        TrainerCallback,
    )
    from trl import AutoModelForCausalLMWithValueHead

    from ...hparams import FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


def _whiten_per_agent(rewards, eps: float = 1e-8):
    """FIX 3: zero-mean / unit-std normalize one agent's reward tensor.

    Applied independently per agent so agents with larger reward magnitudes do
    not dominate the shared PPO value head. Degenerate cases (n<=1 or ~zero
    variance) are mean-centered only to avoid NaNs. Returns a torch tensor.
    """
    if rewards is None or rewards.numel() <= 1:
        return rewards
    std = rewards.std()
    if not torch.isfinite(std) or std < eps:
        return rewards - rewards.mean()
    return (rewards - rewards.mean()) / (std + eps)


# ---------------------------------------------------------------------------
# Joint Seeker+Synthesizer end-to-end reward (opt-in: GRPO_JOINT_E2E=1).
#
# Motivation, from the measured GRPO V6 failure: the default seeker reward
# (recall + 0.3*precision - 0.2*max(0,n-10)/10) has a dead over-selection
# penalty (needs n>10 from a 10-candidate pool) so its only live gradient is
# +0.3*precision, maximised at n=1. The seeker collapsed from 3.5 to 1.2 pages
# within 25 steps, costing 12.5pp of Seeker Recall@all while Recall@1 stayed
# flat -- i.e. it learned nothing about ranking, only to stop hedging.
#
# The e2e evidence shape does not have that gradient. Because
# 1.0*recall + 0.3*precision >= 1.0 whenever recall == 1, the [-1,1] clamp makes
# the reward FLAT at 1.0 across 1..k_budget selected pages, so shrinking buys
# nothing; the over-selection penalty is measured against k_budget=6 (reachable,
# unlike 10); and missing gold costs -0.3 instead of 0.0, widening the hit/miss
# spread that GRPO actually learns from.
#
# Per-agent weights rather than one shared scalar. R_e2e = w_ev*evidence +
# w_ans*answer_f1 depends on both agents, but the synthesizer cannot influence
# the evidence term at all -- and answer_f1 is zero in ~78% of episodes, so a
# shared scalar would hand the synthesizer an advantage driven purely by the
# seeker's sampled page choices on most of its gradient budget. The synthesizer
# therefore gets the answer side only (equivalent to w_ev=0), which keeps
# co-adaptation without the noise channel.
_JOINT_E2E = os.environ.get("GRPO_JOINT_E2E", "0") == "1"
_JOINT_W_EV = float(os.environ.get("VIDORAG_E2E_W_EVIDENCE", "0.4"))
_JOINT_W_ANS = float(os.environ.get("VIDORAG_E2E_W_ANSWER", "0.6"))
_JOINT_K_BUDGET = int(os.environ.get("VIDORAG_E2E_K_BUDGET", "6"))
_JOINT_SYNTH_OVERLAP = os.environ.get("GRPO_JOINT_SYNTH_OVERLAP", "1") == "1"

# KL regularization for GRPO (opt-in: GRPO_KL_BETA > 0).
# Subtracts beta * KL(pi_current || pi_ref) from each rollout's reward before
# computing group-relative advantages. Prevents irreversible policy drift
# past the optimum (observed: V7-JOINT peaked step 39 then regressed 80 steps).
# pi_ref = the warm-start SFT adapter (frozen). Per-agent: each agent's KL is
# computed against its own reference.
_GRPO_KL_BETA = float(os.environ.get("GRPO_KL_BETA", "0.0"))
_GRPO_KL_REF_SEEKER = os.environ.get("GRPO_KL_REF_SEEKER", "")
_GRPO_KL_REF_SYNTHESIZER = os.environ.get("GRPO_KL_REF_SYNTHESIZER", "")

# Phase 1 oracle retrieval bypass: when enabled, the Seeker's generation is
# skipped and gold_pages are used directly as the selection. This ensures the
# Synthesizer trains on oracle-quality retrieval during the curriculum Phase 1.
_PHASE1_ORACLE = os.environ.get("GRPO_PHASE1_ORACLE", "0") == "1"


def _evidence_reward_from_indices(
    selected: List[int], gold: List[int], k_budget: int = _JOINT_K_BUDGET
) -> float:
    """``compute_seeker_evidence_reward`` evaluated on ALREADY-PARSED indices.

    The library version in ``workflow_restore.py`` takes the seeker's raw text
    and expects JSON ``{"choice": [...]}``. This trainer drives the seeker with
    ``SeekerConstrainedLogitsProcessor``, which emits ``Page 3, Page 7`` -- not
    JSON. Feeding that text to the library version falls through to its regex
    fallback and sets ``valid_json=False``, applying a -0.2 penalty on EVERY
    episode. So the formula is reproduced here over ``parse_page_selection``
    output, with the invalid-JSON term dropped as inapplicable.
    """
    if not gold:
        return 0.0
    pred_set, gold_set = set(selected), set(gold)
    hits = len(pred_set & gold_set)
    recall = hits / max(len(gold_set), 1)
    precision = hits / max(len(pred_set), 1) if pred_set else 0.0
    over_selection = -0.05 * max(0, len(pred_set) - k_budget)
    missing_any_gold = -0.3 if hits < len(gold_set) else 0.0
    return max(-1.0, min(1.0, recall + 0.3 * precision + over_selection + missing_any_gold))


# Minimum group std used when normalizing GRPO advantages. 0.0 preserves the
# original behaviour (divide by the raw group std), which is scale-INVARIANT and
# therefore erases the designed weights of a reward: whatever term happens to
# differentiate rollouts inside a group becomes the entire gradient, no matter
# how small it is in absolute terms. Observed concretely on SlideVQA -- a group
# where every rollout found gold but selected 1/2/4/10 pages has std ~0.03, so a
# 0.06 raw width penalty is amplified to advantage -1.50, exactly as strong as
# the -1.50 given to a rollout that missed gold entirely. Three different seeker
# rewards (V6 precision, V7-JOINT answer_f1, V7-BUDGET width penalty) all
# collapsed selection to ~1.3 pages for this reason. A floor of ~0.25 keeps full
# advantage magnitude for groups containing a real hit/miss split (std ~0.5)
# while leaving trivial within-group differences trivial -- hence the default.
# Set MAPPO_GRPO_STD_FLOOR=0.0 to restore the original scale-invariant
# behaviour.
_GRPO_STD_FLOOR = float(os.environ.get("MAPPO_GRPO_STD_FLOOR", "0.25"))


def _grpo_group_advantages(rewards, group_ids, eps: float = 1e-6, std_floor: float = None):
    """GRPO: compute group-relative advantages, NO value head / critic.

    For each group (all samples sharing the same prompt id), advantage =
    (reward - group_mean) / (max(group_std, std_floor) + eps). This replaces
    PPO's value-function baseline, which was the broken component in MAPPO
    (val_var_explained << 0 -> approxkl = 0). Groups of size 1 get advantage 0
    (no relative signal). `rewards` is a 1-D float tensor; `group_ids` is a list
    of hashable ids aligning to `rewards`. Returns a tensor of the same shape.

    `std_floor` defaults to ``MAPPO_GRPO_STD_FLOOR`` (0.25; set 0.0 for the
    original behaviour). See the note on ``_GRPO_STD_FLOOR`` for why it matters.
    """
    import collections
    if std_floor is None:
        std_floor = _GRPO_STD_FLOOR
    adv = rewards.new_zeros(rewards.shape)
    buckets = collections.defaultdict(list)
    for i, g in enumerate(group_ids):
        buckets[g].append(i)
    for g, idxs in buckets.items():
        if len(idxs) <= 1:
            continue  # singleton group -> no relative signal -> advantage 0
        sel = rewards[idxs]
        mean = sel.mean()
        std = sel.std()
        if not torch.isfinite(std):
            centered = sel - mean
        else:
            denom = max(float(std), std_floor)
            if denom < eps:
                centered = sel - mean
            else:
                centered = (sel - mean) / (denom + eps)
        for j, i in enumerate(idxs):
            adv[i] = centered[j]
    return adv


# ---------------------------------------------------------------------------
# Reward interface (imported from vidorag_rewards.py created in Task 4)
# ---------------------------------------------------------------------------
try:
    from .vidorag_rewards import (
        compute_seeker_reward,
        compute_inspector_reward,
        compute_synthesizer_reward,
        compute_batch_rewards,
        REWARD_METHOD,
        VMMOA_ALPHA,
        VMMOA_BETA,
        VMMOA_LAMBDA,
    )
except ImportError:
    logger.warning(
        "vidorag_rewards module not found. Using placeholder reward functions. "
        "Task 4 must create vidorag_rewards.py with the expected interface."
    )
    REWARD_METHOD = "original"
    VMMOA_ALPHA = 0.5
    VMMOA_BETA = 0.3
    VMMOA_LAMBDA = 0.05

    def compute_seeker_reward(predictions: List[str], gold_pages: List[List[int]]) -> torch.Tensor:
        """Placeholder: compare predicted page selections against gold pages."""
        rewards = []
        for pred, gold in zip(predictions, gold_pages):
            try:
                pred_pages = [int(x.strip()) for x in pred.replace("Page", "").split(",") if x.strip().isdigit()]
            except Exception:
                pred_pages = []
            if len(gold) == 0:
                rewards.append(0.0)
                continue
            hits = len(set(pred_pages) & set(gold))
            precision = hits / max(len(pred_pages), 1)
            recall = hits / len(gold)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            rewards.append(f1)
        return torch.tensor(rewards).view(-1, 1)

    def compute_inspector_reward(
        predictions: List[str], gold_routing: List[str], gold_answer: List[str]
    ) -> torch.Tensor:
        """Placeholder: reward for routing decisions."""
        rewards = []
        for pred, gold_r in zip(predictions, gold_routing):
            pred_clean = pred.strip().lower()
            gold_clean = gold_r.strip().lower()
            rewards.append(1.0 if pred_clean == gold_clean else 0.0)
        return torch.tensor(rewards).view(-1, 1)

    def compute_synthesizer_reward(predictions: List[str], gold_answer: List[str]) -> torch.Tensor:
        """Placeholder: F1-based reward for answer generation."""
        rewards = []
        for pred, gold in zip(predictions, gold_answer):
            pred_tokens = pred.lower().split()
            gold_tokens = gold.lower().split()
            common = Counter(pred_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())
            precision = num_same / max(len(pred_tokens), 1)
            recall = num_same / max(len(gold_tokens), 1)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            rewards.append(f1)
        return torch.tensor(rewards).view(-1, 1)


# ---------------------------------------------------------------------------
# MAPPO Independent-Adapter Reward Interface (from vidorag_rewards_restore)
# ---------------------------------------------------------------------------
# These functions implement the per-agent distinct reward system that
# matches the last successful MAPPO run.  Each agent gets its own reward
# computed from its own action; no broadcasting.
# ---------------------------------------------------------------------------
try:
    from .vidorag_rewards_restore import (
        compose_mappo_rewards,
        compute_f1 as _mappo_compute_f1,
        compute_em as _mappo_compute_em,
        normalize_answer as _mappo_normalize_answer,
        parse_json_output as _mappo_parse_json,
        extract_page_indices as _mappo_extract_pages,
        MAPPO_COOP_BONUS,
        MAPPO_COOP_THRESHOLD,
    )
    _MAPPO_RESTORE_AVAILABLE = True
except ImportError:
    logger.warning(
        "vidorag_rewards_restore module not found. "
        "MAPPO independent-adapter rewards will fall back to compute_batch_rewards."
    )
    _MAPPO_RESTORE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class AllowedTokensLogitsProcessor(LogitsProcessor):
    """Constrains generation to only allowed token IDs."""

    def __init__(self, allowed_token_ids: List[int]):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = scores.device
        mask = torch.full(scores.shape, float("-inf"), device=device)
        mask[:, list(self.allowed_token_ids)] = 0
        scores = scores + mask
        return scores


class SeekerConstrainedLogitsProcessor(LogitsProcessor):
    """
    Stateful logits processor that constrains seeker output to the format
    'Page X, Page Y, ...' where X, Y are single digits in [0, num_candidates).

    The Qwen tokenizer splits 'Page 0' into three tokens: ['Page', ' ', '0'].
    So we need a state machine that allows the space token after 'Page'.

    Key constraint: after a digit token, ONLY comma or EOS is allowed.
    This prevents multi-digit numbers like 'Page 27'.
    """

    def __init__(self, tokenizer, num_candidates: int = 10):
        self.tokenizer = tokenizer
        self.num_candidates = num_candidates
        self.eos_id = tokenizer.eos_token_id

        # Build the set of ALL valid token IDs from format strings.
        # This includes 'Page', 'page', ' Page', ' page', ' ', digits, ','.
        self.all_valid_ids: set = set()
        for fmt in [
            "Page 0, Page 1, Page 2, Page 3, Page 4, Page 5, Page 6, Page 7, Page 8, Page 9",
            "page 0, page 1, page 2, page 3, page 4, page 5, page 6, page 7, page 8, page 9",
        ]:
            ids = tokenizer.encode(fmt, add_special_tokens=False)
            self.all_valid_ids.update(ids)
        self.all_valid_ids.add(self.eos_id)

        # Identify single-digit token IDs (e.g. 15..24 for Qwen)
        self.digit_ids: set = set()
        for d in range(10):
            ids = tokenizer.encode(str(d), add_special_tokens=False)
            if len(ids) == 1:
                self.digit_ids.add(ids[0])

        # Comma token IDs
        self.comma_ids: set = set()
        for c in [",", " ,", ", ", " , "]:
            ids = tokenizer.encode(c, add_special_tokens=False)
            for tid in ids:
                decoded = tokenizer.decode([tid]).strip()
                if decoded == ",":
                    self.comma_ids.add(tid)

        # After a digit: only comma or EOS (prevents multi-digit numbers)
        self.after_digit_ids = self.comma_ids | {self.eos_id}

        # Enforce the candidate range. The format strings above contain digits
        # 0-9, so without this the processor lets the seeker emit e.g. 'Page 7'
        # for a 5-candidate episode. parse_page_selection then silently drops
        # that index, which can leave an EMPTY selection and a zero reward the
        # policy cannot learn from. Retrieval + GMM legitimately returns 5-10
        # candidates, so the range has to follow the episode, not a constant.
        self.out_of_range_digit_ids: set = set()
        for d in range(10):
            if d >= num_candidates:
                ids = tokenizer.encode(str(d), add_special_tokens=False)
                if len(ids) == 1:
                    self.out_of_range_digit_ids.add(ids[0])
        self.all_valid_ids -= self.out_of_range_digit_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        last_token_id = input_ids[0, -1].item()

        if last_token_id in self.digit_ids:
            # After a digit: ONLY comma or EOS — prevents 'Page 27'
            allowed = self.after_digit_ids
        else:
            # All other states: allow any valid token from the format strings
            allowed = self.all_valid_ids

        mask = torch.full(scores.shape, float("-inf"), device=scores.device)
        for tid in allowed:
            mask[:, tid] = 0
        return scores + mask


class ImageCache:
    """Simple LRU-like cache for processed image features to avoid recomputation."""

    def __init__(self, max_size: int = 500):
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
            evict_key = self._access_order.pop(0)
            del self._cache[evict_key]
        self._cache[key] = value
        self._access_order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()


# ---------------------------------------------------------------------------
# Main VLM MAPPO Trainer for ViDoRAG
# ---------------------------------------------------------------------------

class ViDoRAGVLMPPOTrainer(PPOTrainer, Trainer):
    r"""
    VLM-adapted MAPPO trainer for ViDoRAG tasks with Qwen2.5-VL-7B.

    Implements a three-agent PPO loop:
      1. Seeker: Given query + candidate page images → generate page selection
      2. Inspector: Given query + selected page images → generate routing decision
      3. Synthesizer: Given query + evidence images → generate answer

    Supports multi-modal inputs (images) alongside text through Qwen2-VL's
    vision processing pipeline (pixel_values, image_grid_thw).
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[List["TrainerCallback"]],
        model: "AutoModelForCausalLMWithValueHead",
        reward_model: Optional["AutoModelForCausalLMWithValueHead"],
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"],
        data_collator: "DataCollatorWithPadding",
        train_dataset: Optional["Dataset"] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> None:
        if eval_dataset is not None:
            raise NotImplementedError("PPOTrainer does not support eval dataset yet.")

        backward_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
        )

        # Add deepspeed config
        if training_args.deepspeed_plugin is not None:
            ppo_config.accelerator_kwargs["kwargs_handlers"] = [
                DistributedDataParallelKwargs(find_unused_parameters=training_args.ddp_find_unused_parameters)
            ]
            ppo_config.accelerator_kwargs["deepspeed_plugin"] = training_args.deepspeed_plugin
            if ppo_config.log_with is not None:
                logger.warning("PPOTrainer cannot use external logger when DeepSpeed is enabled.")
                ppo_config.log_with = None

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * math.ceil(
                len(train_dataset) / total_train_batch_size
            )

        # NOTE: The MA-PPO agent adapters (seeker/inspector/synthesizer) are
        # loaded LATER in _load_mappo_adapters() (called from ppo_train). The
        # optimizer built here therefore does NOT contain their LoRA params, so
        # optimizer.step() silently no-ops on them (grads computed, weights
        # frozen -> approxkl 0, flat eval). _load_mappo_adapters() must rebuild
        # the optimizer afterwards; we keep the training-step count for that.
        self._num_train_steps = num_training_steps

        optimizer = self.create_optimizer(model, training_args, finetuning_args)
        scheduler = self.create_scheduler(training_args, num_training_steps, optimizer)

        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=train_dataset,
            optimizer=optimizer,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.processor = processor
        self.processing_class = tokenizer  # Required by transformers>=5.9 Trainer._save()
        self.current_device = get_current_device()

        # Build eos_token_id list: main eos + any additional special token ids.
        _eos_ids = [self.tokenizer.eos_token_id]
        if hasattr(self.tokenizer, "additional_special_tokens_ids"):
            _eos_ids += self.tokenizer.additional_special_tokens_ids
        elif hasattr(self.tokenizer, "additional_special_tokens"):
            _eos_ids += self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.additional_special_tokens
            )
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=_eos_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        callbacks = DEFAULT_CALLBACKS if callbacks is None else DEFAULT_CALLBACKS + callbacks
        self.callback_handler = CallbackHandler(
            callbacks, self.accelerator.unwrap_model(self.model), self.tokenizer, self.optimizer, self.lr_scheduler
        )
        if self.args.max_steps > 0:
            logger.info("max_steps is given, it will override any value given in num_train_epochs")

        self.amp_context = torch.autocast(self.current_device.type)
        warnings.simplefilter("ignore")

        self.add_callback(FixValueHeadModelCallback)
        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # Batch sizes for 3-agent loop
        self.batch_size_1 = self.config.batch_size
        self.batch_size_2 = 2 * self.batch_size_1
        self.batch_size_3 = 3 * self.batch_size_1

        # Image feature cache
        self.image_cache = ImageCache(max_size=1000)

        # KL-anchored GRPO: record which frozen reference adapter each agent is
        # regularized towards (no-op unless GRPO_KL_BETA > 0).
        self._load_kl_reference_adapters()

    # -----------------------------------------------------------------------
    # Vision processing utilities
    # -----------------------------------------------------------------------

    def load_and_process_images(self, image_paths: List[str]) -> Tuple[List[Image.Image], Dict[str, torch.Tensor]]:
        """
        Load images from paths and process them using Qwen2-VL's image processor.

        Returns:
            pil_images: List of PIL Image objects
            mm_inputs: Dict with 'pixel_values' and 'image_grid_thw' tensors
        """
        pil_images = []
        for path in image_paths:
            if not os.path.exists(path):
                logger.warning(f"Image not found: {path}, using blank placeholder")
                pil_images.append(Image.new("RGB", (224, 224), color=(255, 255, 255)))
            else:
                img = Image.open(path).convert("RGB")
                pil_images.append(img)

        if self.processor is None:
            raise ValueError("Processor is required for VLM training but was not provided.")

        image_processor = getattr(self.processor, "image_processor")
        # Resize images for efficiency
        image_resolution = getattr(self.processor, "image_resolution", 512)
        processed_images = []
        for img in pil_images:
            if max(img.width, img.height) > image_resolution:
                resize_factor = image_resolution / max(img.width, img.height)
                width = int(img.width * resize_factor)
                height = int(img.height * resize_factor)
                img = img.resize((width, height), resample=Image.NEAREST)
            # Qwen2-VL requires minimum dimension of 28
            if min(img.width, img.height) < 28:
                width, height = max(img.width, 28), max(img.height, 28)
                img = img.resize((width, height), resample=Image.NEAREST)
            processed_images.append(img)

        mm_inputs = image_processor(images=processed_images, return_tensors="pt")
        return processed_images, mm_inputs

    def get_cached_image_inputs(self, image_paths: List[str]) -> Dict[str, torch.Tensor]:
        """Load image inputs with caching to avoid reprocessing."""
        cache_key = "|".join(image_paths)  # preserve insertion order, not sorted
        cached = self.image_cache.get(cache_key)
        if cached is not None:
            return cached

        _, mm_inputs = self.load_and_process_images(image_paths)
        self.image_cache.put(cache_key, mm_inputs)
        return mm_inputs

    def build_vlm_messages_with_images(
        self, role: str, query: str, image_paths: List[str], system_prompt: str
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """
        Build chat messages with image placeholders for Qwen2-VL.

        Uses <|vision_start|><|image_pad|>...<|vision_end|> tokens.
        Returns the messages list and the image paths used.
        """
        # Build image placeholder string
        image_placeholders = ""
        for _i in range(len(image_paths)):
            image_placeholders += "<|vision_start|><|image_pad|><|vision_end|>\n"

        user_content = f"{image_placeholders}\n{query}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        return messages, image_paths

    # -----------------------------------------------------------------------
    # Agent prompt construction
    # -----------------------------------------------------------------------

    def get_seeker_messages(
        self, question: str, candidate_image_paths: List[str]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Build messages for Seeker agent: select relevant pages from candidates."""
        num_pages = len(candidate_image_paths)
        system_prompt = (
            f"You are a document page selection assistant. Given a question and {num_pages} candidate "
            f"document page images (Page 0 to Page {num_pages - 1}), select the pages that are most "
            "relevant to answering the question. Output selected page numbers separated by commas, "
            "e.g., 'Page 0, Page 3, Page 5'."
        )
        query = f"Question: {question}\n\nThe following are {num_pages} candidate document pages:"
        return self.build_vlm_messages_with_images("seeker", query, candidate_image_paths, system_prompt)

    def get_inspector_messages(
        self, question: str, selected_image_paths: List[str]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Build messages for Inspector agent: decide routing (answer directly or retrieve more).

        Uses v6 JSON format to match Inspector SFT v6 / PPO v1 adapter training:
          Output: {"sufficient": true/false, "answer": <str or null>,
                   "reference": [0-based indices], "missing": [hints]}
        """
        system_prompt = (
            "You are a sufficiency judge. Decide if the given pages are enough to "
            "answer the question. Use candidate indices only. Do not repeat page text. "
            "Return JSON: {\"sufficient\": true/false, \"answer\": <str or null>, "
            "\"reference\": [indices], \"missing\": [hints]}."
        )
        query = (
            f"Question: {question}\n\n"
            f"The following {len(selected_image_paths)} pages have been selected as evidence:"
        )
        return self.build_vlm_messages_with_images("inspector", query, selected_image_paths, system_prompt)

    def get_synthesizer_messages(
        self, question: str, evidence_image_paths: List[str]
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        """Build messages for Synthesizer agent: generate answer from evidence pages."""
        system_prompt = (
            "You are a document question-answering assistant. Given a question and relevant document "
            "page images as evidence, generate a concise and accurate answer. Only output the answer."
        )
        query = (
            f"Question: {question}\n\n"
            f"The following {len(evidence_image_paths)} document pages contain the relevant evidence:"
        )
        return self.build_vlm_messages_with_images("synthesizer", query, evidence_image_paths, system_prompt)

    # -----------------------------------------------------------------------
    # Tokenization with vision inputs
    # -----------------------------------------------------------------------

    def trans_vlm_messages_to_inputs(
        self, messages_list: List[List[Dict[str, str]]], images_list: List[List[str]]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize VLM messages and prepare vision inputs for batch processing.

        Args:
            messages_list: List of chat message sequences (one per sample)
            images_list: List of image path lists (one list per sample)

        Returns:
            Dict containing input_ids, attention_mask, pixel_values, image_grid_thw
        """
        input_ids_list = []
        all_pixel_values = []
        all_image_grid_thw = []

        for messages, image_paths in zip(messages_list, images_list):
            # Tokenize text with chat template
            if self.tokenizer.chat_template is not None:
                # Process images to get mm_inputs for determining token counts
                mm_inputs = self.get_cached_image_inputs(image_paths) if image_paths else {}

                # Build full text with vision tokens
                _result = self.tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                # Handle BatchEncoding (transformers>=5.9) or raw tensor.
                if hasattr(_result, "input_ids"):
                    text = _result.input_ids.to(self.current_device)
                elif hasattr(_result, "to"):
                    text = _result.to(self.current_device)
                else:
                    text = torch.tensor(_result, device=self.current_device)
            else:
                _result = self.tokenizer(
                    self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
                    return_tensors="pt",
                )
                text = _result.input_ids.to(self.current_device)

            # Expand <|image_pad|> tokens to match actual image feature count.
            # In transformers>=5.9 the chat template inserts a single image_pad
            # per image, but the model expects grid_t*grid_h*grid_w/merge_size^2
            # tokens per image.
            if image_paths and mm_inputs and "image_grid_thw" in mm_inputs:
                image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
                merge_size = 2  # Qwen3.5-VL merge_size
                grid_thw = mm_inputs["image_grid_thw"]  # [num_images, 3]
                tokens_per_image = []
                for gi in range(grid_thw.size(0)):
                    t, h, w = grid_thw[gi]
                    tokens_per_image.append(int(t.item() * h.item() * w.item() // (merge_size ** 2)))
                ids_1d = text.squeeze(0)  # [seq_len]
                new_ids = []
                img_idx = 0
                for tid in ids_1d.tolist():
                    if tid == image_pad_id and img_idx < len(tokens_per_image):
                        new_ids.extend([image_pad_id] * tokens_per_image[img_idx])
                        img_idx += 1
                    else:
                        new_ids.append(tid)
                text = torch.tensor([new_ids], device=self.current_device, dtype=text.dtype)

            input_ids_list.append(text.squeeze(0))

            # Collect vision inputs
            if image_paths:
                if "pixel_values" in mm_inputs:
                    all_pixel_values.append(mm_inputs["pixel_values"])
                if "image_grid_thw" in mm_inputs:
                    all_image_grid_thw.append(mm_inputs["image_grid_thw"])

        # Pad input_ids to same length
        max_length = max(ids.size(0) for ids in input_ids_list)
        input_ids_padded = torch.stack([
            torch.cat([
                ids.new_full((max_length - ids.size(0),), self.tokenizer.pad_token_id),
                ids
            ], dim=0)
            for ids in input_ids_list
        ], dim=0)

        attention_masks = torch.stack([
            torch.cat([
                torch.zeros(max_length - ids.size(0), dtype=torch.long),
                torch.ones(ids.size(0), dtype=torch.long)
            ], dim=0)
            for ids in input_ids_list
        ], dim=0).cuda()

        batch = {
            "input_ids": input_ids_padded,
            "attention_mask": attention_masks,
        }

        # Add vision inputs if present
        if all_pixel_values:
            batch["pixel_values"] = torch.cat(all_pixel_values, dim=0).to(self.current_device)
        if all_image_grid_thw:
            batch["image_grid_thw"] = torch.cat(all_image_grid_thw, dim=0).to(self.current_device)
            # Store per-sample thw row counts so _grpo_step can slice correctly
            # when each sample has a different number of images.
            batch["thw_counts"] = [thw.shape[0] for thw in all_image_grid_thw]

        return batch

    # -----------------------------------------------------------------------
    # Generation with vision inputs
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def get_vlm_inputs(
        self,
        batch: Dict[str, torch.Tensor],
        logits_processor: Optional[LogitsProcessorList] = None,
        max_new_tokens: int = 100,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Generate model responses given queries with vision inputs.

        Passes both text tokens and pixel values to model.generate().
        """
        if batch["input_ids"].size(0) == 1:
            start_index = (batch["input_ids"][0] != self.tokenizer.pad_token_id).nonzero()[0].item()
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dim() >= 2 and k in ("input_ids", "attention_mask"):
                    batch[k] = v[:, start_index:]

        with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
            if self.model_args.upcast_layernorm:
                layernorm_params = dump_layernorm(unwrapped_model)

            generate_kwargs = {
                "generation_config": self.generation_config,
                "logits_processor": logits_processor or get_logits_processor(),
                "max_new_tokens": max_new_tokens,
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            # Pass vision inputs for VLM generation
            if "pixel_values" in batch:
                generate_kwargs["pixel_values"] = batch["pixel_values"]
            if "image_grid_thw" in batch:
                generate_kwargs["image_grid_thw"] = batch["image_grid_thw"]

            generate_output = unwrapped_model.generate(**generate_kwargs)

            if self.model_args.upcast_layernorm:
                restore_layernorm(unwrapped_model, layernorm_params)

        query = batch["input_ids"].detach().cpu()
        response = generate_output[:, batch["input_ids"].size(-1):].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            query_start_index = (query[i] != self.tokenizer.pad_token_id).nonzero()[0].item()
            response_indexes = (response[i] != self.tokenizer.pad_token_id).nonzero()

            if len(response_indexes) == 0:
                response_length = 1
            elif self.tokenizer.eos_token_id == self.tokenizer.pad_token_id:
                response_length = response_indexes[-1].item() + 2
            else:
                response_length = response_indexes[-1].item() + 1

            queries.append(query[i, query_start_index:])
            responses.append(response[i, :response_length])

        return queries, responses

    # -----------------------------------------------------------------------
    # Retrieval integration
    # -----------------------------------------------------------------------

    def fetch_candidate_pages(self, questions: List[str], top_n: int = 10) -> List[List[str]]:
        """
        Call Flask retrieval server to get candidate page image paths.

        Args:
            questions: List of query strings
            top_n: Number of candidate pages to retrieve

        Returns:
            List of lists of image file paths
        """
        current_gpu = torch.cuda.current_device()
        url = f"http://127.0.0.1:800{current_gpu}/search"
        data = {"questions": questions, "N": top_n}

        try:
            response = requests.post(url, json=data, timeout=30)
            payload = response.json()
            all_page_paths = []
            for result in payload.get("results", []):
                docs = result.get("documents", [])
                page_paths = []
                for d in docs:
                    img_path = d.get("image_path", "")
                    if not img_path:
                        continue
                    # Normalise double-path bug from retrieval server
                    # (e.g. .../data/MMLongBench/data/MMLongBench/img/ -> .../data/MMLongBench/img/)
                    img_path = img_path.replace(
                        "/data/MMLongBench/data/MMLongBench/img/",
                        "/data/MMLongBench/img/",
                    )
                    # Same fix for SlideVQA
                    img_path = img_path.replace(
                        "/data/SlideVQA/data/SlideVQA/img/",
                        "/data/SlideVQA/img/",
                    )
                    page_paths.append(img_path)
                all_page_paths.append(page_paths[:top_n])
            return all_page_paths
        except Exception as e:
            logger.warning(f"Retrieval server request failed: {e}. Returning empty results.")
            return [[] for _ in questions]

    # -----------------------------------------------------------------------
    # Page selection parsing
    # -----------------------------------------------------------------------

    def parse_page_selection(self, answer_text: str, num_candidates: int) -> List[int]:
        """Parse seeker output to extract selected page indices."""
        # Match patterns like "Page 0, Page 3, Page 5" or "0, 3, 5" or "Page0,Page3"
        numbers = re.findall(r"\d+", answer_text)
        selected = []
        for n in numbers:
            idx = int(n)
            if 0 <= idx < num_candidates and idx not in selected:
                selected.append(idx)
        return selected[:num_candidates]  # cap at max candidates

    def get_seeker_format_reward(self, answer_text: str, num_candidates: int) -> float:
        """Penalize malformed seeker output."""
        pattern = r"^(Page\s*\d+\s*,\s*)*(Page\s*\d+)$"
        if not re.match(pattern, answer_text.strip(), re.IGNORECASE):
            return -0.5
        numbers = re.findall(r"\d+", answer_text)
        if len(numbers) != len(set(numbers)):
            return -0.5
        for n in numbers:
            if int(n) >= num_candidates:
                return -0.5
        return 0.0

    # -----------------------------------------------------------------------
    # Main PPO training loop
    # -----------------------------------------------------------------------

    def _load_mappo_adapters(self) -> None:
        """Load 3 independent LoRA adapters for MAPPO independent-adapter mode.

        Each adapter is warm-started from its per-agent PPO checkpoint:
          - seeker adapter from VIDORAG_SEEKER_ADAPTER
          - inspector adapter from VIDORAG_INSPECTOR_ADAPTER
          - synthesizer adapter from VIDORAG_SYNTHESIZER_ADAPTER

        The base model is shared; only the active LoRA adapter differs per agent.
        All adapters are set to trainable so PPO can update them independently.
        """
        import os as _os

        adapter_paths = {
            "seeker": _os.environ.get("VIDORAG_SEEKER_ADAPTER", ""),
            "inspector": _os.environ.get("VIDORAG_INSPECTOR_ADAPTER", ""),
            "synthesizer": _os.environ.get("VIDORAG_SYNTHESIZER_ADAPTER", ""),
        }

        unwrapped = self.accelerator.unwrap_model(self.model)
        peft_model = getattr(unwrapped, 'pretrained_model', unwrapped)

        # The default adapter (loaded from adapter_name_or_path in YAML)
        # will be used as the base. We load the other two as additional adapters.
        # If all three paths are provided, we load them as named adapters.
        loaded_adapters = []
        for agent_name, adapter_path in adapter_paths.items():
            # Fail fast: silently substituting the default adapter would make
            # several agents share one adapter while the run is still labeled
            # independent-agent MAPPO/GRPO.
            if not adapter_path or not _os.path.isdir(adapter_path):
                raise FileNotFoundError(
                    f"MAPPO independent requires a valid {agent_name} adapter directory "
                    f"(VIDORAG_{agent_name.upper()}_ADAPTER), got: {adapter_path!r}"
                )

            try:
                # Load adapter with agent-specific name
                peft_model.load_adapter(adapter_path, adapter_name=agent_name, is_trainable=True)
                loaded_adapters.append((agent_name, agent_name))
                logger.info(f"MAPPO independent: Loaded {agent_name} adapter from {adapter_path}")
            except Exception as e:
                raise RuntimeError(
                    f"MAPPO independent failed to load {agent_name} adapter from {adapter_path}"
                ) from e

        self._adapter_names = [name for _, name in loaded_adapters]
        # NOTE: Do NOT call set_adapter() here — PEFT's set_adapter disables
        # requires_grad for non-active adapters, which would exclude the
        # inspector/synthesizer LoRA params from the optimizer rebuild below.
        # The active adapter is set later, after the optimizer is rebuilt.

        # CRITICAL FIX (v4): PEFT's load_adapter(is_trainable=True) may NOT
        # actually set requires_grad=True for all loaded adapters in all PEFT
        # versions. Explicitly enable requires_grad for ALL LoRA params across
        # ALL adapters so the optimizer includes every adapter's weights.
        self._enable_all_adapter_grads(peft_model)
        _n_trainable_before = sum(1 for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"MAPPO independent: after load_adapter + _enable_all_adapter_grads: "
            f"{_n_trainable_before} trainable param tensors"
        )

        # GAP 3 FIX: Per-agent value heads. A single shared value head cannot
        # model 3 agents with different reward distributions (R_ret, R_ins,
        # R_ans), leading to catastrophic val_var_explained (-80 to -125) even
        # with warm-start from a single checkpoint. Create 3 separate value
        # head weight stores, each warm-started from its agent's PPO checkpoint.
        # Before each self.step() call, swap the model's v_head weights to the
        # per-agent weights; after step, save the updated weights back. The
        # optimizer sees the same v_head parameters (same tensor objects), so
        # param groups remain valid — only the weight values change per agent.
        self._per_agent_v_heads = {0: None, 1: None, 2: None}
        _agent_names_vh = ["seeker", "inspector", "synthesizer"]
        if hasattr(unwrapped, "v_head"):
            import safetensors.torch as _st
            for _a_idx, _a_name in enumerate(_agent_names_vh):
                _vh_path = _os.path.join(adapter_paths.get(_a_name, ""),
                                         "value_head.safetensors")
                if _vh_path and _os.path.exists(_vh_path):
                    try:
                        _vh_sd = _st.load_file(_vh_path)
                        self._per_agent_v_heads[_a_idx] = _vh_sd
                        logger.info(f"MAPPO independent: Loaded {_a_name} value head from {_vh_path}")
                    except Exception as e:
                        logger.warning(f"MAPPO independent: failed to load {_a_name} value head: {e}")
                        self._per_agent_v_heads[_a_idx] = {
                            k: v.clone() for k, v in unwrapped.v_head.state_dict().items()
                        }
                else:
                    logger.warning(
                        f"MAPPO independent: {_a_name} value_head.safetensors not found, "
                        f"using current v_head weights as fallback"
                    )
                    self._per_agent_v_heads[_a_idx] = {
                        k: v.clone() for k, v in unwrapped.v_head.state_dict().items()
                    }

            # Load seeker's v_head into the model as the initial active one
            if self._per_agent_v_heads[0] is not None:
                unwrapped.v_head.load_state_dict(self._per_agent_v_heads[0])
                logger.info("MAPPO independent: Set seeker value head as active (initial)")
        else:
            logger.warning("MAPPO independent: model has no v_head attribute, per-agent value heads disabled")

        # ROOT CAUSE FIX (0729): The optimizer was created in __init__() BEFORE
        # these three agent LoRA adapters existed, so self.optimizer's param
        # groups do not contain the adapter weights. _grpo_step()/step()
        # computed correct gradients and called optimizer.step(), but Adam had
        # no knowledge of the adapter params -> policy never moved (approxkl=0,
        # eval flat at the warm-start score). Rebuild optimizer + scheduler now
        # that all trainable adapter params are loaded. Safe for the plain-
        # accelerator single-GPU case; deepspeed/fsdp rebuilds its own engine
        # from these anyway.
        try:
            logger.info(
                "MAPPO independent: rebuilding optimizer to include the "
                f"{len(loaded_adapters)} agent adapter(s) (created before adapters "
                "were loaded, so the old optimizer excluded their LoRA params)."
            )
            # Re-enable grads right before rebuild (in case any upstream
            # code path disabled them after our earlier call).
            self._enable_all_adapter_grads(peft_model)
            optimizer = self.create_optimizer(self.model, self.args, self.finetuning_args)
            scheduler = self.create_scheduler(self.args, self._num_train_steps, optimizer)
            self.optimizer = optimizer
            self.lr_scheduler = scheduler
            trainable = sum(1 for p in self.model.parameters() if p.requires_grad)
            opt_params = sum(1 for g in self.optimizer.param_groups for _ in g["params"])
            logger.info(
                f"MAPPO independent: optimizer rebuilt over {opt_params} params "
                f"({trainable} trainable)."
            )
        except Exception as e:  # pragma: no cover
            logger.warning(f"MAPPO independent: optimizer rebuild failed: {e}")

        # NOW set the active adapter (after optimizer rebuild so all adapters
        # are included). set_adapter may toggle requires_grad for non-active
        # adapters, but _set_agent_adapter() re-enables requires_grad for ALL
        # adapter params on every call, so the optimizer always sees them.
        if self._adapter_names:
            peft_model.set_adapter(self._adapter_names[0])
            self._enable_all_adapter_grads(peft_model)
            logger.info(f"MAPPO independent: Active adapter set to {self._adapter_names[0]} (seeker)")

    def _swap_v_head(self, agent_idx: int) -> None:
        """Swap the model's v_head weights to the per-agent value head.

        Called before self.step() to ensure the correct value head is active
        for the current agent's PPO update. This implements per-agent critics
        without modifying TRL's optimizer or model structure — only the weight
        values are swapped in-place.
        """
        if not self._mappo_independent or not hasattr(self, '_per_agent_v_heads'):
            return
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "v_head") and self._per_agent_v_heads.get(agent_idx) is not None:
            unwrapped.v_head.load_state_dict(self._per_agent_v_heads[agent_idx])

    def _save_v_head(self, agent_idx: int) -> None:
        """Save the model's updated v_head weights back to the per-agent store.

        Called after self.step() to persist the PPO-updated value head weights
        for the current agent. The next time this agent is updated, the
        improved weights will be loaded.
        """
        if not self._mappo_independent or not hasattr(self, '_per_agent_v_heads'):
            return
        unwrapped = self.accelerator.unwrap_model(self.model)
        if hasattr(unwrapped, "v_head"):
            self._per_agent_v_heads[agent_idx] = {
                k: v.clone() for k, v in unwrapped.v_head.state_dict().items()
            }

    def _enable_all_adapter_grads(self, peft_model) -> None:
        """Re-enable requires_grad=True for ALL loaded LoRA adapter params.

        PEFT's set_adapter() may disable requires_grad for non-active adapters,
        which excludes them from the optimizer's param groups. This method
        ensures ALL adapter params remain trainable so the optimizer can
        update whichever adapter is currently active.

        Only the ACTIVE adapter is in the forward computation graph, so
        backward() only produces gradients for the active adapter — the
        others' params remain with p.grad=None even though requires_grad=True.

        KL reference adapters (``kl_ref_*``) are excluded: they are the frozen
        pi_ref of KL-anchored GRPO and must never receive gradients.
        """
        _count = 0
        for name, param in peft_model.named_parameters():
            if 'kl_ref_' in name:
                param.requires_grad = False
                continue
            # Match LoRA params across all adapters by name. PEFT stores them
            # as e.g. 'base_model.model.model.layers.0.self_attn.q_proj
            # .lora_A.seeker.weight' — the 'lora_' substring is reliable.
            if 'lora_' in name:
                param.requires_grad = True
                _count += 1
            # Also enable v_head params (value head)
            elif 'v_head' in name or 'value_head' in name:
                param.requires_grad = True
                _count += 1
        logger.debug(
            f"_enable_all_adapter_grads: enabled requires_grad for "
            f"{_count} param tensors"
        )

    def _set_agent_adapter(self, agent_idx: int) -> None:
        """Switch the active LoRA adapter for the given agent.

        Args:
            agent_idx: 0=seeker, 1=inspector, 2=synthesizer
        """
        if not self._mappo_independent or agent_idx >= len(self._adapter_names):
            return
        adapter_name = self._adapter_names[agent_idx]
        unwrapped = self.accelerator.unwrap_model(self.model)
        peft_model = getattr(unwrapped, 'pretrained_model', unwrapped)
        if hasattr(peft_model, 'set_adapter'):
            peft_model.set_adapter(adapter_name)
        # CRITICAL: set_adapter() may disable requires_grad for non-active
        # adapters, which would exclude them from the optimizer. Re-enable
        # requires_grad for ALL adapter params so the optimizer always
        # sees them. Only the active adapter receives gradients during
        # backward (others are not in the computation graph).
        self._enable_all_adapter_grads(peft_model)

    def _load_kl_reference_adapters(self) -> None:
        """Record the frozen reference adapter of each agent for KL-anchored GRPO.

        No-op unless ``GRPO_KL_BETA > 0``. Only the PATHS are stored here: the
        agent adapters themselves are not loaded until ``_load_mappo_adapters()``
        runs inside ``ppo_train()``, so the reference adapters are attached
        lazily on the first KL computation (see ``_compute_kl_penalties``).

        The default reference of each agent is its own warm-start adapter, i.e.
        the policy GRPO starts from, which is what "do not drift past the
        optimum" means in practice. Override per agent with GRPO_KL_REF_SEEKER /
        GRPO_KL_REF_SYNTHESIZER.
        """
        self._kl_ref_paths: Dict[int, str] = {}
        if _GRPO_KL_BETA <= 0:
            return
        seeker_ref = _GRPO_KL_REF_SEEKER or os.environ.get("VIDORAG_SEEKER_ADAPTER", "")
        synth_ref = _GRPO_KL_REF_SYNTHESIZER or os.environ.get("VIDORAG_SYNTHESIZER_ADAPTER", "")
        if seeker_ref:
            self._kl_ref_paths[0] = seeker_ref        # agent 0 = seeker
        if synth_ref:
            self._kl_ref_paths[2] = synth_ref         # agent 2 = synthesizer
        logger.info(
            f"KL-anchored GRPO: beta={_GRPO_KL_BETA}, reference adapters: "
            f"seeker={seeker_ref!r}, synthesizer={synth_ref!r}"
        )

    def _compute_kl_penalties(self, agent_idx: int, batches) -> Optional[List[float]]:
        """Per-sequence KL(pi_current || pi_ref) for KL-anchored GRPO.

        Uses the standard RLHF approximation KL ~ mean(log pi_current - log
        pi_ref) over the RESPONSE tokens only (the same "approx KL" TRL logs).
        The prompt tokens are excluded because the policy has no control over
        them, so including them would just add a constant-ish offset that the
        group-relative baseline would cancel anyway.

        `batches` is ``agent_batches[agent_idx]``: a list of
        ``(batch_dict, queries, responses)`` captured at rollout time. The
        returned list is in the SAME flat order the samples were appended, so it
        aligns element-wise with the agent's reward tensor. Returns None when no
        reference is configured or the computation fails (KL penalty is then
        skipped rather than corrupting the rewards).
        """
        import torch.nn.functional as F

        ref_path = getattr(self, "_kl_ref_paths", {}).get(agent_idx, "")
        if not ref_path or not self._mappo_independent:
            return None
        if agent_idx >= len(self._adapter_names):
            return None

        device = self.current_device
        current_adapter = self._adapter_names[agent_idx]
        ref_adapter_name = f"kl_ref_{agent_idx}"
        unwrapped = self.accelerator.unwrap_model(self.model)
        peft_model = getattr(unwrapped, "pretrained_model", unwrapped)

        # Attach the reference adapter once, frozen. It is loaded AFTER the
        # optimizer rebuild in _load_mappo_adapters(), so its params are not in
        # any param group and can never be stepped.
        try:
            if ref_adapter_name not in getattr(peft_model, "peft_config", {}):
                peft_model.load_adapter(ref_path, adapter_name=ref_adapter_name,
                                        is_trainable=False)
                for name, param in peft_model.named_parameters():
                    if ref_adapter_name in name:
                        param.requires_grad = False
                logger.info(
                    f"KL reference adapter {ref_adapter_name} loaded from {ref_path}"
                )
        except Exception as e:
            logger.warning(
                f"KL reference adapter load failed for agent {agent_idx}: {e}. "
                "Skipping KL penalty."
            )
            self._set_agent_adapter(agent_idx)
            return None

        kl_values: List[float] = []
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                for (batch_dict, q_list, r_list) in batches:
                    for row in range(len(q_list)):
                        q = q_list[row].to(device)
                        r = r_list[row].to(device)
                        input_ids = torch.cat([q, r], dim=0).unsqueeze(0)
                        attn = torch.ones_like(input_ids)
                        fwd = dict(input_ids=input_ids, attention_mask=attn,
                                   return_dict=True, use_cache=False)
                        pv, thw = self._row_mm_inputs(batch_dict, row, len(q_list))
                        if pv is not None and thw is not None:
                            fwd["pixel_values"] = pv
                            fwd["image_grid_thw"] = thw

                        resp_len = r.shape[0]
                        targets = input_ids[:, 1:]

                        def _resp_logp(adapter: str):
                            peft_model.set_adapter(adapter)
                            out = self.model(**fwd)
                            logits = out.logits if hasattr(out, "logits") else out[0]
                            logp_all = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
                            tok = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
                            return tok[-resp_len:]

                        cur_logp = _resp_logp(current_adapter)
                        ref_logp = _resp_logp(ref_adapter_name)
                        kl_values.append(float((cur_logp - ref_logp).mean()))
        except Exception as e:
            logger.warning(
                f"KL computation failed for agent {agent_idx}: {e}. Skipping KL penalty."
            )
            kl_values = None
        finally:
            # Always put the trainable adapter back (and re-enable its grads,
            # which set_adapter may have cleared) before the GRPO update.
            self._set_agent_adapter(agent_idx)
            if was_training:
                self.model.train()

        return kl_values

    def _row_mm_inputs(self, batch_dict, row: int, n_rows: int):
        """Slice one rollout row's (pixel_values, image_grid_thw) out of a batch.

        Same per-row slicing ``_grpo_step`` performs, factored out so the KL
        reference forward pass sees exactly the images the policy forward sees.
        Returns (None, None) when the batch carries no images.
        """
        device = self.current_device
        try:
            pv = batch_dict.get("pixel_values") if isinstance(batch_dict, dict) else None
            thw = batch_dict.get("image_grid_thw") if isinstance(batch_dict, dict) else None
        except Exception:
            return None, None
        if pv is None or thw is None:
            return None, None
        try:
            thw_counts = batch_dict.get("thw_counts") if isinstance(batch_dict, dict) else None
            per_thw_patches = [int(t.prod().item()) for t in thw]
            if thw_counts is not None and row < len(thw_counts):
                start_thw = sum(thw_counts[:row])
                end_thw = start_thw + thw_counts[row]
            elif thw.dim() == 2 and thw.shape[0] == n_rows:
                start_thw, end_thw = row, row + 1
            else:
                return pv.to(device), thw.to(device)
            row_thw = thw[start_thw:end_thw].to(device)
            startp = sum(per_thw_patches[:start_thw])
            patches = sum(per_thw_patches[start_thw:end_thw])
            return pv[startp:startp + patches].to(device), row_thw
        except Exception:
            return pv.to(device), thw.to(device)

    def _oracle_seeker_outputs(self, batch, oracle_texts: List[str]):
        """Phase 1 oracle bypass: build the seeker's "response" from gold pages.

        Returns ``(queries, responses)`` in exactly the format
        ``get_vlm_inputs`` returns, so the rest of the episode (downstream
        agents, rewards, GRPO batch capture) is unchanged — only the seeker's
        generation call is skipped. The gold pages are candidate INDICES, so the
        text matches the ``Page X, Page Y`` format the constrained seeker emits
        and ``parse_page_selection`` round-trips it.
        """
        query = batch["input_ids"].detach().cpu()
        queries, responses = [], []
        for i in range(len(query)):
            nz = (query[i] != self.tokenizer.pad_token_id).nonzero()
            start_index = nz[0].item() if len(nz) else 0
            queries.append(query[i, start_index:])
            text = oracle_texts[i] if i < len(oracle_texts) else ""
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                ids = [self.tokenizer.eos_token_id]
            responses.append(torch.tensor(ids, dtype=query.dtype))
        return queries, responses

    def _grpo_step(self, batch_records, advantages):
        """GRPO policy-gradient update for the ACTIVE adapter (no value head).

        Critic-free replacement for TRL's PPO self.step(): maximises
        E[ A * mean_token_logprob(response | query) ] with precomputed
        GROUP-RELATIVE advantages. No value head, so the MAPPO failure mode
        (val_var_explained << 0 -> approxkl = 0) cannot occur.

        CRITICAL (bug fixed): the forward pass MUST include the multimodal inputs
        (pixel_values / image_grid_thw) captured at rollout time. The earlier
        version forwarded input_ids ONLY, so for Qwen2.5-VL the <image> tokens
        were treated as plain text, the response log-probs were meaningless, and
        the vision-side LoRA received no gradient. Net effect: healthy advantages
        (mean_abs_adv ~ 0.5) but the policy never moved (approxkl=0, score flat).

        Args:
            batch_records: list of (batch_dict, queries, responses) captured per
                rollout mini-batch for THIS agent. `batch_dict` carries
                input_ids/attention_mask and, when present, pixel_values /
                image_grid_thw. `queries`/`responses` are lists of 1-D LongTensors
                aligned to the batch rows.
            advantages: flat list/tensor of per-sample advantages, in the SAME
                global order the samples were appended for this agent.
        Returns a stats dict for logging.
        """
        import torch.nn.functional as F

        device = self.current_device
        model = self.model
        model.train()
        clip = float(os.environ.get("MAPPO_GRPO_ADV_CLIP", "5.0"))

        # advantages are indexed globally across all this agent's samples in the
        # order they were appended (matches how agent_group_ids was built).
        adv_list = [float(a) for a in advantages]
        if clip > 0:
            adv_list = [max(-clip, min(clip, a)) for a in adv_list]

        # Gradient normalization. Every sample calls .backward() separately, so
        # gradients ACCUMULATE (sum) across the whole agent slice. Without this
        # divisor the effective gradient scales with batch_size * group_size
        # (64 samples at bs=16, G=4), which is what produced max|grad| in the
        # thousands and moved the policy far off the warm-start in one step.
        # Dividing each sample loss by the number of contributing samples makes
        # the accumulated gradient a MEAN, so the step size is independent of
        # batch size and group size.
        n_active = sum(1 for a in adv_list if a != 0.0) or 1
        total_loss = 0.0
        updated = 0
        self.optimizer.zero_grad()

        global_i = 0
        for (batch_dict, q_list, r_list) in batch_records:
            for row in range(len(q_list)):
                adv = adv_list[global_i] if global_i < len(adv_list) else 0.0
                global_i += 1
                if adv == 0.0:
                    continue  # singleton/constant group -> no signal

                q = q_list[row].to(device)
                r = r_list[row].to(device)
                input_ids = torch.cat([q, r], dim=0).unsqueeze(0)
                attn = torch.ones_like(input_ids)

                # Build the multimodal forward kwargs from the rollout batch so
                # the <image> tokens are backed by real pixel features and the
                # vision LoRA gets gradient. Slice the row's images via
                # image_grid_thw if the batch was multi-row.
                fwd = dict(input_ids=input_ids, attention_mask=attn,
                           return_dict=True, use_cache=False)
                try:
                    pv = batch_dict.get("pixel_values") if isinstance(batch_dict, dict) else None
                    thw = batch_dict.get("image_grid_thw") if isinstance(batch_dict, dict) else None
                except Exception:
                    pv, thw = None, None
                if pv is not None and thw is not None:
                    # Per-row image slice: each sample may have multiple images,
                    # so we use thw_counts (per-sample image counts) to locate
                    # this row's image_grid_thw rows and pixel_values block.
                    try:
                        thw_counts = batch_dict.get("thw_counts") if isinstance(batch_dict, dict) else None
                        if thw_counts is not None and row < len(thw_counts):
                            # Slice thw: rows [start_thw:end_thw] belong to this sample
                            start_thw = sum(thw_counts[:row])
                            end_thw = start_thw + thw_counts[row]
                            row_thw = thw[start_thw:end_thw].to(device)
                            # pixel_values are concatenated in the same order as
                            # image_grid_thw rows; each thw row contributes
                            # prod(t,h,w) patches.
                            per_thw_patches = [int(t.prod().item()) for t in thw]
                            patches = sum(per_thw_patches[start_thw:end_thw])
                            startp = sum(per_thw_patches[:start_thw])
                            row_pv = pv[startp:startp + patches].to(device)
                        elif thw.dim() == 2 and thw.shape[0] == len(q_list):
                            # Fallback: 1 image per sample (thw rows align 1:1)
                            row_thw = thw[row:row + 1].to(device)
                            patches = int(row_thw.prod(dim=-1).sum().item())
                            per_row = [int(t.prod().item()) for t in thw]
                            startp = sum(per_row[:row])
                            row_pv = pv[startp:startp + patches].to(device)
                        else:
                            row_thw = thw.to(device)
                            row_pv = pv.to(device)
                        fwd["pixel_values"] = row_pv
                        fwd["image_grid_thw"] = row_thw
                    except Exception:
                        # If slicing fails, fall back to whole-batch tensors.
                        fwd["pixel_values"] = pv.to(device)
                        fwd["image_grid_thw"] = thw.to(device)

                out = model(**fwd)
                logits = out.logits if hasattr(out, "logits") else out[0]
                logits = logits[:, :-1, :]
                targets = input_ids[:, 1:]
                logp_all = F.log_softmax(logits.float(), dim=-1)
                tok_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]
                resp_len = r.shape[0]
                resp_logp = tok_logp[-resp_len:]
                sample_loss = -(adv * resp_logp.mean())
                # Backprop the normalized loss, but report the raw per-sample
                # loss so the logged value stays comparable across batch sizes.
                (sample_loss / n_active).backward()
                total_loss += float(sample_loss.detach())
                updated += 1

        # 0729 ROOT CAUSE VERIFY: log grad magnitude before optimizer.step()
        # to confirm the rebuilt optimizer covers adapter params. If max|grad|=0
        # the optimizer still excludes the agent adapters (rebuild failed).
        # V4 FIX: also log total optimizer params vs params with grad to detect
        # the inspector-zero-gradient bug (optimizer only had seeker params).
        try:
            import torch as _t
            if _t.is_grad_enabled():
                _total_params = sum(len(g["params"]) for g in self.optimizer.param_groups)
                _grads = [p.grad.abs().sum().item()
                          for g in self.optimizer.param_groups
                          for p in g["params"] if p.grad is not None]
                _n_with_grad = len(_grads)
                _max_grad = max(_grads) if _grads else 0.0
                logger.info(
                    f"GRPO check: {_n_with_grad}/{_total_params} params with grad, "
                    f"max |grad| = {_max_grad:.6f}"
                )
        except Exception:
            pass

        # Bound the update. The GRPO path bypasses TRL's step(), which is where
        # clipping normally happens, so without this a single high-advantage
        # sample can move a well-tuned warm-start policy far enough to cause the
        # catastrophic regressions seen in the earlier GRPO runs.
        grad_norm = 0.0
        max_grad_norm = float(getattr(self.args, "max_grad_norm", 0.0) or 0.0)
        if max_grad_norm > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"]],
                max_grad_norm,
            ))

        self.optimizer.step()
        # Advance the LR schedule. TRL's step() normally does this; the GRPO
        # path replaces it, so without this call the schedule stays pinned at
        # step 0 forever. With lr_scheduler_type=constant that is harmless, but
        # any warmup schedule would sit at its step-0 multiplier of 0, i.e. a
        # learning rate of exactly 0 and no learning at all.
        if getattr(self, "lr_scheduler", None) is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

        try:
            cur_lr = (self.lr_scheduler.get_last_lr()[0]
                      if getattr(self, "lr_scheduler", None) is not None
                      else float(self.optimizer.param_groups[0]["lr"]))
        except Exception:
            cur_lr = 0.0

        mean_abs_adv = (sum(abs(a) for a in adv_list) / len(adv_list)) if adv_list else 0.0
        mean_loss = total_loss / max(1, updated)
        return {
            "grpo/loss": mean_loss,
            "grpo/updated_samples": updated,
            "grpo/mean_abs_adv": mean_abs_adv,
            "grpo/grad_norm": grad_norm,
            "ppo/loss/total": mean_loss,  # keep key for downstream logging
            "ppo/learning_rate": cur_lr,
        }

    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        """
        Implements the VLM MAPPO training loop for ViDoRAG.

        Three-agent loop per step:
          1. Seeker: query + candidate images → page selection
          2. Inspector: query + selected images → routing decision
          3. Synthesizer: query + evidence images → answer generation
        """
        start_step = 0
        _resume_ckpt = resume_from_checkpoint or os.environ.get("VIDORAG_RESUME_CHECKPOINT", "")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running ViDoRAG VLM MAPPO Training *****")
            logger.info("  Reward method = {}".format(REWARD_METHOD))
            if REWARD_METHOD == "vmmoa":
                logger.info(
                    "  V-MMOA coefficients: alpha(R_ret)={}, beta(R_ins)={}, lambda(C)={}".format(
                        VMMOA_ALPHA, VMMOA_BETA, VMMOA_LAMBDA
                    )
                )
            logger.info("  Num examples = {:,}".format(num_examples))
            logger.info("  Num Epochs = {:,}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {:,}".format(self.args.per_device_train_batch_size))
            logger.info(
                "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {:,}".format(
                    total_train_batch_size
                )
            )
            logger.info("  Gradient Accumulation steps = {:,}".format(self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {:,}".format(self.finetuning_args.ppo_epochs))
            logger.info("  Total training steps = {:,}".format(max_steps))
            logger.info("  Number of trainable parameters = {:,}".format(count_parameters(self.model)[0]))

        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.callback_handler.on_train_begin(self.args, self.state, self.control)

        # Load gold answers for reward computation
        answers_path = os.path.join(
            self.args.output_dir, "..", "vidorag_gold_answers.json"
        )
        questions_golden_answers_dict = self._load_gold_answers(answers_path)

        # ---- Held-out evaluation set for periodic eval during training ----
        eval_holdout_size = int(os.environ.get("VIDORAG_EVAL_HOLDOUT_SIZE", "80"))
        eval_steps_interval = int(os.environ.get("VIDORAG_EVAL_STEPS", "25"))
        eval_questions: List[str] = []
        eval_gold_data: List[Dict[str, Any]] = []
        if questions_golden_answers_dict:
            _all_qs = list(questions_golden_answers_dict.keys())
            random.seed(42)
            random.shuffle(_all_qs)
            eval_questions = _all_qs[:eval_holdout_size]
            eval_gold_data = [questions_golden_answers_dict[q] for q in eval_questions]
            logger.info(
                "  Held-out eval set: {} samples, eval every {} steps".format(
                    len(eval_questions), eval_steps_interval
                )
            )

        # Seeker constrained logits processor: stateful processor that
        # enforces 'Page X, Page Y' format with single-digit indices only.
        # One processor per candidate count, because the valid index range is
        # 0..len(candidates)-1 and retrieval returns 5-10 candidates. A single
        # processor pinned at 10 would permit indices the episode does not have.
        _seeker_lp_cache: Dict[int, LogitsProcessorList] = {}

        def _seeker_lp(num_candidates: int) -> LogitsProcessorList:
            n = max(1, min(10, int(num_candidates)))
            if n not in _seeker_lp_cache:
                _seeker_lp_cache[n] = LogitsProcessorList([
                    SeekerConstrainedLogitsProcessor(self.tokenizer, num_candidates=n)
                ])
            return _seeker_lp_cache[n]

        # Default used by the periodic eval path, which fetches 10 candidates.
        seeker_logits_processor = _seeker_lp(10)

        # ---- MAPPO independent-adapter support ----
        # When VIDORAG_MAPPO_INDEPENDENT=1, load 3 separate LoRA adapters
        # (seeker, inspector, synthesizer) and switch between them during
        # each agent's forward pass. Each adapter is warm-started from its
        # per-agent PPO checkpoint.
        self._mappo_independent = os.environ.get("VIDORAG_MAPPO_INDEPENDENT", "0") == "1"
        self._adapter_names = ["default"]  # Start with the default loaded adapter
        if self._mappo_independent:
            self._load_mappo_adapters()

        # Logging
        kl_ctl_path = os.path.join(self.args.output_dir, "kl_ctl.txt")
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(kl_ctl_path, "a") as f:
            f.write(f"init_kl_coef: {self.config.init_kl_coef}, target: {self.config.target}\n\n")

        self.config.ppo_epochs = 2

        # ---- Resume from checkpoint ----
        if _resume_ckpt and os.path.isdir(_resume_ckpt):
            import safetensors.torch as st
            logger.info(f"Resuming from checkpoint: {_resume_ckpt}")
            unwrapped = self.accelerator.unwrap_model(self.model)
            # Load LoRA adapter
            _adapter_file = os.path.join(_resume_ckpt, "adapter_model.safetensors")
            if os.path.exists(_adapter_file) and hasattr(unwrapped, 'pretrained_model'):
                unwrapped.pretrained_model.load_adapter(_resume_ckpt, "default", is_trainable=True)
                unwrapped.pretrained_model.set_adapter("default")
                logger.info(f"Loaded LoRA adapter from {_resume_ckpt}")
            # Load value head
            _vhead_file = os.path.join(_resume_ckpt, "value_head.safetensors")
            if os.path.exists(_vhead_file) and hasattr(unwrapped, 'v_head'):
                _vhead_state = st.load_file(_vhead_file)
                unwrapped.v_head.load_state_dict(_vhead_state)
                logger.info(f"Loaded value head from {_vhead_file}")
            # Extract step number from checkpoint directory name (checkpoint-50 -> 50)
            _ckpt_name = os.path.basename(_resume_ckpt.rstrip("/"))
            try:
                start_step = int(_ckpt_name.split("-")[-1])
                self.state.global_step = start_step
                logger.info(f"Resuming from step {start_step}, global_step={start_step}")
            except ValueError:
                logger.warning(f"Could not parse step from {_ckpt_name}, starting from 0")
            # Fast-forward dataloader past already-trained batches
            for _ in range(start_step):
                try:
                    next(dataiter)
                except StopIteration:
                    dataiter = iter(self.dataloader)
                    next(dataiter)
            logger.info(f"Dataloader fast-forwarded {start_step} steps")

        for step in tqdm(range(start_step, max_steps), disable=not self.is_local_process_zero()):
            with open(kl_ctl_path, "a") as f:
                f.write(f"step: {step}, kl_ctl.value: {self.kl_ctl.value}\n")

            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Convert BatchEncoding to plain dict for robust slicing (transformers>=5.9).
            if not isinstance(batch, dict):
                batch = {k: v for k, v in batch.items() if v is not None}

            # ---- Rollout phase ----
            self.model.eval()
            self.tokenizer.padding_side = "right"
            queries, responses, rewards = [], [], []

            # GRPO: enable grouped sampling so each prompt yields G distinct
            # responses (forming a non-singleton group for advantage
            # computation). Without this, _grpo_group_advantages sees all
            # singleton groups and every advantage is 0 (no learning).
            _use_grpo = os.environ.get("MAPPO_USE_GRPO", "0") == "1"
            _grpo_g = int(os.environ.get("MAPPO_GRPO_GROUP_SIZE", "4"))
            if _use_grpo and _grpo_g > 1:
                # Patch generation_config to enable sampling for diverse responses
                self.generation_config.do_sample = True
                self.generation_config.temperature = float(
                    os.environ.get("MAPPO_GRPO_TEMPERATURE", "0.8"))
                self.generation_config.top_p = float(
                    os.environ.get("MAPPO_GRPO_TOP_P", "0.95"))
                logger.info(
                    f"GRPO grouped sampling: G={_grpo_g}, "
                    f"temperature={self.generation_config.temperature}, "
                    f"top_p={self.generation_config.top_p}"
                )
            elif _use_grpo:
                logger.warning(
                    "GRPO active but MAPPO_GRPO_GROUP_SIZE<=1: groups will be "
                    "singletons -> all advantages 0 -> no learning. Set "
                    "MAPPO_GRPO_GROUP_SIZE>=4."
                )

            # FIX P1: keep each agent's transitions separate so the PPO update
            # can be applied to that agent's OWN adapter. A single shared
            # self.step() over the concatenated batch backpropagates every
            # agent's data into whichever adapter happens to be active last
            # (the synthesizer), defeating the "independent adapter" design.
            agent_queries = {0: [], 1: [], 2: []}    # 0=seeker,1=inspector,2=synth
            agent_responses = {0: [], 1: [], 2: []}
            agent_rewards = {0: [], 1: [], 2: []}
            agent_group_ids = {0: [], 1: [], 2: []}  # GRPO: prompt id per sample
            # GRPO image-forward fix: keep the per-agent tokenized rollout batch
            # (which carries pixel_values / image_grid_thw) and the generated
            # responses so the GRPO update can recompute log-probs via the SAME
            # multimodal forward used for generation. Without images the VLM
            # forward is text-only, the <image> placeholder log-probs are
            # meaningless, and the vision LoRA gets no gradient -> policy never
            # moves (approxkl=0) despite healthy advantages.
            agent_batches = {0: [], 1: [], 2: []}  # list of (batch_dict, queries, responses)

            # Always optimize all 3 agents
            self.config.batch_size = self.batch_size_3
            torch.cuda.empty_cache()

            for idx in range(0, self.batch_size_1, self.config.mini_batch_size):
                mini_batch = {k: v[idx: idx + self.config.mini_batch_size] for k, v in batch.items()}

                # Extract questions and associated data from the batch
                mini_batch_questions = []
                mini_batch_gold_answers = []
                mini_batch_gold_routing = []
                mini_batch_gold_docs = []
                mini_batch_reference_pages = []

                for sub_idx in range(self.config.mini_batch_size):
                    init_text = self.tokenizer.decode(mini_batch["input_ids"][sub_idx], skip_special_tokens=True)
                    # Parse question from the init_text
                    question = self._extract_question_from_text(init_text)
                    mini_batch_questions.append(question)

                    # Get gold data
                    gold_data = questions_golden_answers_dict.get(question.strip(), {})
                    mini_batch_gold_answers.append(gold_data.get("answer", ""))
                    mini_batch_gold_routing.append(gold_data.get("routing", "SUFFICIENT"))
                    mini_batch_gold_docs.append(gold_data.get("gold_doc", ""))
                    mini_batch_reference_pages.append(gold_data.get("reference_pages", gold_data.get("gold_pages", [])))

                # ==================== GRPO: Expand prompts G times ====================
                # When GRPO is active, replicate each prompt G times so the
                # rollout produces G distinct responses per question (with
                # do_sample=True). All G copies share the same group_id (the
                # question text), giving _grpo_group_advantages real variance.
                # Without this, every group is a singleton -> advantage 0.
                if _use_grpo and _grpo_g > 1:
                    mini_batch_questions = [
                        q for q in mini_batch_questions for _ in range(_grpo_g)
                    ]
                    mini_batch_gold_answers = [
                        a for a in mini_batch_gold_answers for _ in range(_grpo_g)
                    ]
                    mini_batch_gold_routing = [
                        r for r in mini_batch_gold_routing for _ in range(_grpo_g)
                    ]
                    mini_batch_gold_docs = [
                        d for d in mini_batch_gold_docs for _ in range(_grpo_g)
                    ]
                    mini_batch_reference_pages = [
                        p for p in mini_batch_reference_pages for _ in range(_grpo_g)
                    ]

                # ==================== SEEKER ====================
                # Switch to seeker adapter for independent-adapter MAPPO
                self._set_agent_adapter(0)  # 0 = seeker

                # Use dataset images as candidates (NOT retrieval server).
                # The MAPPO episode dataset already has 10 candidate images per sample,
                # with gold pages guaranteed to be among them. Fetching from the
                # retrieval server returns pages from different documents, causing
                # gold_candidate_idx to always be empty.
                candidate_pages = []
                for i in range(len(mini_batch_questions)):
                    gold_data = questions_golden_answers_dict.get(mini_batch_questions[i].strip(), {})
                    dataset_images = gold_data.get("image_paths", [])
                    if dataset_images:
                        candidate_pages.append(dataset_images)
                    else:
                        # Fallback: fetch from retrieval server
                        fetched = self.fetch_candidate_pages([mini_batch_questions[i]], top_n=10)
                        candidate_pages.append(fetched[0] if fetched else [])

                # The gold answers file stores gold_pages as candidate INDICES
                # (e.g., [6] means candidate #6 is the gold page), NOT as
                # document page numbers. Use them directly — no doc/page matching.
                mini_batch_gold_pages = []
                for i in range(len(mini_batch_questions)):
                    gold_data = questions_golden_answers_dict.get(mini_batch_questions[i].strip(), {})
                    gold_candidate_indices = gold_data.get("gold_pages", [])
                    # Validate indices are in range
                    num_cands = len(candidate_pages[i])
                    valid_indices = [idx for idx in gold_candidate_indices
                                      if isinstance(idx, int) and 0 <= idx < num_cands]
                    mini_batch_gold_pages.append(valid_indices)

                seeker_messages_list = []
                seeker_images_list = []
                for i, question in enumerate(mini_batch_questions):
                    msgs, imgs = self.get_seeker_messages(question, candidate_pages[i])
                    seeker_messages_list.append(msgs)
                    seeker_images_list.append(imgs)

                # Tokenize and generate. Constrain the index range to the
                # SHORTEST candidate list in the mini-batch so every emitted
                # index is valid for every row (with mini_batch_size=1 and GRPO
                # group replication all rows share one candidate list anyway).
                seeker_batch = self.trans_vlm_messages_to_inputs(seeker_messages_list, seeker_images_list)
                _n_cands = min((len(c) for c in candidate_pages if c), default=10)
                if _PHASE1_ORACLE:
                    # Phase 1 curriculum: skip the seeker's generation and hand
                    # the gold pages downstream, so the inspector/synthesizer
                    # train on oracle-quality retrieval instead of whatever the
                    # not-yet-trained seeker happened to sample. The seeker's
                    # own transitions are still recorded (its update should be
                    # disabled with GRPO_FREEZE_SEEKER=1 during Phase 1).
                    _oracle_texts = [
                        "Page " + ", Page ".join(str(p) for p in mini_batch_gold_pages[i])
                        if mini_batch_gold_pages[i] else "Page 0"
                        for i in range(len(mini_batch_questions))
                    ]
                    if step == start_step:
                        logger.info(
                            "GRPO_PHASE1_ORACLE=1: seeker generation bypassed, "
                            f"using gold pages (e.g. {_oracle_texts[0]!r})"
                        )
                    seeker_queries, seeker_responses = self._oracle_seeker_outputs(
                        seeker_batch, _oracle_texts
                    )
                else:
                    seeker_queries, seeker_responses = self.get_vlm_inputs(
                        seeker_batch, logits_processor=_seeker_lp(_n_cands), max_new_tokens=50
                    )

                # Decode seeker answers
                seeker_answers_text = []
                mini_batch_selected_pages = []
                for i, resp in enumerate(seeker_responses):
                    text = self.tokenizer.decode(resp, skip_special_tokens=True)
                    seeker_answers_text.append(text)
                    num_cands = len(candidate_pages[i])
                    selected = self.parse_page_selection(text, num_cands)
                    mini_batch_selected_pages.append(selected)

                queries.extend(seeker_queries)
                responses.extend(seeker_responses)
                agent_queries[0].extend(seeker_queries)
                agent_responses[0].extend(seeker_responses)
                # GRPO: keep the tokenized batch (with pixel_values) + responses
                agent_batches[0].append((seeker_batch, seeker_queries, seeker_responses))

                # ==================== INSPECTOR ====================
                # Switch to inspector adapter for independent-adapter MAPPO
                self._set_agent_adapter(1)  # 1 = inspector

                inspector_messages_list = []
                inspector_images_list = []
                for i, question in enumerate(mini_batch_questions):
                    # Get selected page image paths
                    selected_imgs = [
                        candidate_pages[i][idx]
                        for idx in mini_batch_selected_pages[i]
                        if idx < len(candidate_pages[i])
                    ]
                    if not selected_imgs:
                        # Fallback: use first candidate if selection empty
                        selected_imgs = candidate_pages[i][:1] if candidate_pages[i] else []
                    msgs, imgs = self.get_inspector_messages(question, selected_imgs)
                    inspector_messages_list.append(msgs)
                    inspector_images_list.append(imgs)

                inspector_batch = self.trans_vlm_messages_to_inputs(inspector_messages_list, inspector_images_list)
                inspector_queries, inspector_responses = self.get_vlm_inputs(
                    inspector_batch, max_new_tokens=100
                )

                inspector_answers_text = []
                for resp in inspector_responses:
                    text = self.tokenizer.decode(resp, skip_special_tokens=True)
                    inspector_answers_text.append(text)

                queries.extend(inspector_queries)
                responses.extend(inspector_responses)
                agent_queries[1].extend(inspector_queries)
                agent_responses[1].extend(inspector_responses)
                agent_batches[1].append((inspector_batch, inspector_queries, inspector_responses))

                # ==================== SYNTHESIZER ====================
                # Switch to synthesizer adapter for independent-adapter MAPPO
                self._set_agent_adapter(2)  # 2 = synthesizer

                synthesizer_messages_list = []
                synthesizer_images_list = []
                for i, question in enumerate(mini_batch_questions):
                    evidence_imgs = [
                        candidate_pages[i][idx]
                        for idx in mini_batch_selected_pages[i]
                        if idx < len(candidate_pages[i])
                    ]
                    if not evidence_imgs:
                        evidence_imgs = candidate_pages[i][:1] if candidate_pages[i] else []
                    msgs, imgs = self.get_synthesizer_messages(question, evidence_imgs)
                    synthesizer_messages_list.append(msgs)
                    synthesizer_images_list.append(imgs)

                synthesizer_batch = self.trans_vlm_messages_to_inputs(
                    synthesizer_messages_list, synthesizer_images_list
                )
                synthesizer_queries, synthesizer_responses = self.get_vlm_inputs(
                    synthesizer_batch, max_new_tokens=150
                )

                synthesizer_answers_text = []
                for resp in synthesizer_responses:
                    text = self.tokenizer.decode(resp, skip_special_tokens=True)
                    synthesizer_answers_text.append(text)

                queries.extend(synthesizer_queries)
                responses.extend(synthesizer_responses)
                agent_queries[2].extend(synthesizer_queries)
                agent_responses[2].extend(synthesizer_responses)
                agent_batches[2].append((synthesizer_batch, synthesizer_queries, synthesizer_responses))

                # Debug: log actual agent outputs for the first few steps
                if step < 5:
                    for dbg_i in range(len(mini_batch_questions)):
                        _sel = self.parse_page_selection(seeker_answers_text[dbg_i], len(candidate_pages[dbg_i]))
                        _cand_info = []
                        for _ci, _cp in enumerate(candidate_pages[dbg_i][:3]):
                            _did, _pn = self._extract_doc_page_from_path(_cp)
                            _cand_info.append(f"  cand[{_ci}]: doc={_did} page={_pn} path=...{_cp[-60:]}")
                        print(
                            f"[DEBUG step={step}] Q={mini_batch_questions[dbg_i][:60]}...\n"
                            f"  seeker_out={repr(seeker_answers_text[dbg_i][:200])}\n"
                            f"  seeker_parsed={_sel}  gold_candidate_idx={mini_batch_gold_pages[dbg_i]}  num_cands={len(candidate_pages[dbg_i])}\n"
                            f"  gold_doc={repr(mini_batch_gold_docs[dbg_i])}  ref_pages={mini_batch_reference_pages[dbg_i]}\n"
                            f"  {chr(10).join(_cand_info)}\n"
                            f"  inspector_out={repr(inspector_answers_text[dbg_i][:200])}\n"
                            f"  synthesizer_out={repr(synthesizer_answers_text[dbg_i][:200])}\n"
                            f"  gold_answer={repr(mini_batch_gold_answers[dbg_i][:100])}",
                            flush=True,
                        )

                # ==================== REWARDS ====================
                # Use compose_mappo_rewards (per-agent distinct rewards) to match
                # the last successful MAPPO run's behavior.
                s_rw, i_rw, y_rw, _details = self._compute_mappo_rewards_batch(
                    seeker_outputs=seeker_answers_text,
                    inspector_outputs=inspector_answers_text,
                    synthesizer_outputs=synthesizer_answers_text,
                    gold_answers=mini_batch_gold_answers,
                    gold_pages=mini_batch_gold_pages,
                    selected_pages=mini_batch_selected_pages,
                    step=step,
                    is_eval=False,
                )
                seeker_rewards = torch.tensor(s_rw, dtype=torch.float).to(self.current_device)
                inspector_rewards = torch.tensor(i_rw, dtype=torch.float).to(self.current_device)
                synthesizer_rewards = torch.tensor(y_rw, dtype=torch.float).to(self.current_device)

                # Add seeker format penalties
                seeker_format_penalties = torch.tensor([
                    self.get_seeker_format_reward(t, len(candidate_pages[i]))
                    for i, t in enumerate(seeker_answers_text)
                ], dtype=torch.float).to(self.current_device)
                seeker_total_reward = seeker_rewards + seeker_format_penalties

                # FIX 3: per-agent reward whitening. DEFAULT OFF now: the last run
                # showed whitening collapsed returns_var to ~0.04-0.10, which drove
                # approxkl -> 0 (policy froze) and contributed to degradation. Keep
                # it opt-in for experiments. Only whiten when the batch is large
                # enough that mean/std are meaningful (n>=MIN_WHITEN); on tiny
                # batches whitening turns the sign into noise.
                MIN_WHITEN = int(os.environ.get("MAPPO_MIN_WHITEN", "8"))
                if os.environ.get("MAPPO_WHITEN_PER_AGENT", "0") == "1":
                    if seeker_total_reward.numel() >= MIN_WHITEN:
                        seeker_total_reward = _whiten_per_agent(seeker_total_reward)
                    if inspector_rewards.numel() >= MIN_WHITEN:
                        inspector_rewards = _whiten_per_agent(inspector_rewards)
                    if synthesizer_rewards.numel() >= MIN_WHITEN:
                        synthesizer_rewards = _whiten_per_agent(synthesizer_rewards)

                rewards.extend(seeker_total_reward)
                rewards.extend(inspector_rewards)
                rewards.extend(synthesizer_rewards)

                # FIX P1: also record each agent's rewards separately so the PPO
                # update can be applied to that agent's own adapter below.
                agent_rewards[0].extend(list(seeker_total_reward))
                agent_rewards[1].extend(list(inspector_rewards))
                agent_rewards[2].extend(list(synthesizer_rewards))
                # GRPO: record the group id (prompt) for each sample so advantages
                # can be normalized within the group of responses to the same
                # question. Same prompt order for all three agents in this loop.
                for _q in mini_batch_questions:
                    agent_group_ids[0].append(_q)
                    agent_group_ids[1].append(_q)
                    agent_group_ids[2].append(_q)

                # Log rewards (post-whitening values actually used for PPO)
                self._log_step_rewards(step, seeker_total_reward, inspector_rewards, synthesizer_rewards)

            # ---- PPO update phase ----
            self.model.train()
            _use_grpo = os.environ.get("MAPPO_USE_GRPO", "0") == "1"
            if self._mappo_independent and _use_grpo:
                # GRPO: critic-free, group-relative advantage policy gradient.
                # Removes the value head entirely (the component that broke MAPPO:
                # val_var_explained << 0 -> approxkl = 0). Advantages are
                # normalized within the group of responses to the same prompt, and
                # each agent's own adapter is updated on its own transitions.
                per_agent_stats = {}
                _FREEZE_ENV = {
                    0: "GRPO_FREEZE_SEEKER",
                    1: "GRPO_FREEZE_INSPECTOR",
                    2: "GRPO_FREEZE_SYNTHESIZER",
                }
                for a_idx in (0, 1, 2):
                    # A frozen agent still runs in the rollout (so the pipeline
                    # stays end-to-end and its outputs still feed the other
                    # agents' rewards) but receives no gradients.
                    #
                    # V6 froze the synthesizer: the warm-start synthesizer
                    # (PPO v11) is already well-tuned, and updating it with GRPO
                    # caused 31/300 samples to regress (score 5->≤2) from the
                    # distribution shift as the seeker changed.
                    #
                    # The joint Seeker+Synthesizer experiment freezes the
                    # INSPECTOR instead -- it is not a bottleneck (oracle skip
                    # cost -0.07) and its SFT-v6 routing is adequate -- so that
                    # seeker and synthesizer can co-adapt.
                    _fenv = _FREEZE_ENV[a_idx]
                    if os.environ.get(_fenv, "0") == "1":
                        logger.info(f"{_fenv}=1: skipping agent {a_idx} update")
                        per_agent_stats[a_idx] = {
                            "grpo/loss": 0.0,
                            "grpo/updated_samples": 0,
                            "grpo/mean_abs_adv": 0.0,
                            "grpo/grad_norm": 0.0,
                            "ppo/loss/total": 0.0,
                            # Without this key the top-level log reports
                            # learning_rate=0.0 whenever this agent is the one
                            # selected for `stats` below, which looks exactly
                            # like a broken LR schedule.
                            "ppo/learning_rate": float(
                                self.lr_scheduler.get_last_lr()[0]
                                if getattr(self, "lr_scheduler", None) is not None
                                else 0.0
                            ),
                        }
                        continue
                    a_q = agent_queries[a_idx]
                    a_r = agent_responses[a_idx]
                    a_rw = agent_rewards[a_idx]
                    a_gid = agent_group_ids[a_idx]
                    a_batches = agent_batches[a_idx]
                    if not a_q:
                        continue
                    # Fail fast on singleton groups: a group with one response
                    # has zero variance, so every advantage is 0 and the update
                    # silently becomes a no-op (looks like training, learns
                    # nothing). Requires MAPPO_GRPO_GROUP_SIZE >= 2 replication.
                    from collections import Counter as _Counter
                    _group_counts = _Counter(a_gid)
                    _singletons = [gid for gid, cnt in _group_counts.items() if cnt < 2]
                    if _singletons:
                        raise RuntimeError(
                            "GRPO requires at least two sampled responses per prompt, but "
                            f"agent {a_idx} has {len(_singletons)} singleton group(s) "
                            f"(e.g. {_singletons[0]!r:.80}). Set MAPPO_GRPO_GROUP_SIZE>=2 so "
                            "rollouts are replicated before enabling MAPPO_USE_GRPO."
                        )
                    rw_tensor = torch.stack([r if torch.is_tensor(r) else torch.tensor(float(r))
                                             for r in a_rw]).float().to(self.current_device)
                    # KL anchor (opt-in, GRPO_KL_BETA>0): charge each rollout for
                    # its drift from the warm-start policy BEFORE the group
                    # baseline is subtracted. Applied after normalization it
                    # would be a constant shift per group and cancel out; applied
                    # here it makes a rollout worth its advantage only if it
                    # beats its group by more than the KL it spent.
                    _kl_mean = None
                    if _GRPO_KL_BETA > 0 and a_idx in getattr(self, "_kl_ref_paths", {}):
                        kl_penalties = self._compute_kl_penalties(a_idx, a_batches)
                        if kl_penalties and len(kl_penalties) == rw_tensor.numel():
                            kl_tensor = torch.tensor(kl_penalties, dtype=rw_tensor.dtype,
                                                     device=rw_tensor.device)
                            rw_tensor = rw_tensor - _GRPO_KL_BETA * kl_tensor
                            _kl_mean = float(kl_tensor.mean())
                            logger.info(
                                f"  KL penalty agent {a_idx}: mean={_kl_mean:.4f}, "
                                f"beta*KL mean={_GRPO_KL_BETA * _kl_mean:.4f}"
                            )
                        elif kl_penalties:
                            # Misalignment would subtract one rollout's KL from
                            # another's reward, which is worse than no anchor.
                            logger.warning(
                                f"KL penalty agent {a_idx}: {len(kl_penalties)} KL values for "
                                f"{rw_tensor.numel()} rewards -- misaligned, skipping."
                            )
                    adv = _grpo_group_advantages(rw_tensor, a_gid)
                    self._set_agent_adapter(a_idx)
                    # Pass the captured rollout batches (with pixel_values) so the
                    # GRPO forward includes images (see _grpo_step docstring).
                    per_agent_stats[a_idx] = self._grpo_step(a_batches, adv)
                    if _kl_mean is not None:
                        per_agent_stats[a_idx]["grpo/kl"] = _kl_mean
                # Restore the 3-agent batch size for the next rollout.
                self.config.batch_size = self.batch_size_3
                # Top-level logs should come from an agent that actually
                # updated. Unconditionally preferring the synthesizer makes
                # loss/learning_rate read 0.0 whenever it is frozen, hiding the
                # real training signal behind a dummy stats dict.
                stats = None
                for _cand in (2, 1, 0):
                    _s = per_agent_stats.get(_cand)
                    if _s and _s.get("grpo/updated_samples", 0):
                        stats = _s
                        break
                if stats is None:
                    stats = (per_agent_stats.get(2)
                             or per_agent_stats.get(1)
                             or per_agent_stats.get(0)
                             or {"ppo/loss/total": 0.0})
            elif self._mappo_independent:
                # FIX P1: update each agent's OWN adapter with only that agent's
                # transitions. A single self.step() over the concatenated batch
                # would backprop all three agents' data into whichever adapter is
                # active last (synthesizer), so seeker/inspector never actually
                # learn — defeating independent-adapter MAPPO. Here we activate
                # each adapter in turn and step only on its slice. TRL's step()
                # asserts config.batch_size == len(queries), so set it per call
                # (each agent slice has batch_size_1 items, not batch_size_3).
                per_agent_stats = {}
                for a_idx in (0, 1, 2):
                    a_q = agent_queries[a_idx]
                    a_r = agent_responses[a_idx]
                    a_rw = agent_rewards[a_idx]
                    if not a_q:
                        continue
                    self._set_agent_adapter(a_idx)
                    self._swap_v_head(a_idx)  # GAP 3: load per-agent critic
                    self.config.batch_size = len(a_q)
                    per_agent_stats[a_idx] = self.step(a_q, a_r, a_rw)
                    self._save_v_head(a_idx)  # GAP 3: persist updated critic
                # Restore the 3-agent batch size for the next rollout.
                self.config.batch_size = self.batch_size_3
                # Use the synthesizer's stats (or the last available) for the
                # main logs below. Per-agent approxkl/val_var_explained are
                # logged separately so all three agents are visible.
                stats = (per_agent_stats.get(2)
                         or per_agent_stats.get(1)
                         or per_agent_stats.get(0))
            else:
                stats = self.step(queries, responses, rewards)

            # Ensure minimum KL coefficient
            MIN_BETA = 0.05
            if self.kl_ctl.value < MIN_BETA:
                self.kl_ctl.value = MIN_BETA

            self.tokenizer.padding_side = "left"
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning("Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.callback_handler.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                def _safe_stat(key, default=0.0, src=None):
                    """Safely extract a float from PPO stats (handles tensors/arrays).

                    Args:
                        src: optional stats dict to read from (defaults to the
                            global ``stats`` used for the main logs). Pass
                            per_agent_stats[a_idx] to read a specific agent's
                            stats.
                    """
                    _src = src if src is not None else stats
                    try:
                        val = _src.get(key, default)
                        if isinstance(val, torch.Tensor):
                            return val.float().mean().item()
                        if hasattr(val, "item"):
                            return float(val.item())
                        if hasattr(val, "__iter__") and not isinstance(val, str):
                            return float(sum(val) / len(val)) if len(val) > 0 else 0.0
                        return float(val)
                    except (TypeError, ValueError, IndexError, ZeroDivisionError):
                        return default

                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    policy_loss=round(_safe_stat("ppo/loss/policy"), 6),
                    value_loss=round(_safe_stat("ppo/loss/value"), 6),
                    entropy=round(_safe_stat("ppo/policy/entropy"), 6),
                    approxkl=round(_safe_stat("ppo/policy/approxkl"), 6),
                    clipfrac=round(_safe_stat("ppo/policy/clipfrac"), 4),
                    val_error=round(_safe_stat("ppo/val/error"), 6),
                    val_var_explained=round(_safe_stat("ppo/val/var_explained"), 6),
                    returns_mean=round(_safe_stat("ppo/returns/mean"), 4),
                    returns_var=round(_safe_stat("ppo/returns/var"), 4),
                    kl_coef=round(float(self.kl_ctl.value), 6),
                    learning_rate=_safe_stat("ppo/learning_rate", 0.0),
                    epoch=round(step / steps_in_epoch, 2),
                )

                # GAP 2 FIX: log per-agent PPO diagnostics so all three agents
                # are visible. Without this, only the synthesizer's stats (from
                # the ``stats`` selection above) appear in logs — seeker and
                # inspector approxkl/val_var_explained are silently discarded.
                if self._mappo_independent and per_agent_stats:
                    _agent_names = ["seeker", "inspector", "synthesizer"]
                    for _a_idx, _a_name in enumerate(_agent_names):
                        _a_stats = per_agent_stats.get(_a_idx)
                        if _a_stats is None:
                            continue
                        logs[f"{_a_name}/approxkl"] = round(
                            _safe_stat("ppo/policy/approxkl", src=_a_stats), 6)
                        logs[f"{_a_name}/val_var_explained"] = round(
                            _safe_stat("ppo/val/var_explained", src=_a_stats), 6)
                        logs[f"{_a_name}/value_loss"] = round(
                            _safe_stat("ppo/loss/value", src=_a_stats), 6)
                        logs[f"{_a_name}/entropy"] = round(
                            _safe_stat("ppo/policy/entropy", src=_a_stats), 6)
                        # GRPO-specific per-agent stats (only present when
                        # MAPPO_USE_GRPO=1; otherwise these keys are absent and
                        # _safe_stat returns the default 0.0).
                        logs[f"{_a_name}/grpo_mean_abs_adv"] = round(
                            _safe_stat("grpo/mean_abs_adv", src=_a_stats), 6)
                        logs[f"{_a_name}/grpo_updated_samples"] = int(
                            _safe_stat("grpo/updated_samples", src=_a_stats))
                        logs[f"{_a_name}/grpo_grad_norm"] = round(
                            _safe_stat("grpo/grad_norm", src=_a_stats), 4)
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.callback_handler.on_log(self.args, self.state, self.control, logs)
                loss_meter.reset()
                reward_meter.reset()

            # ---- Periodic held-out evaluation ----
            if (
                self.is_local_process_zero()
                and eval_questions
                and (step + 1) % eval_steps_interval == 0
            ):
                try:
                    # Fixed eval subset: always use the same N samples for
                    # comparable results across checkpoints.
                    _eval_n = min(int(os.environ.get("VIDORAG_EVAL_SIZE", "40")), len(eval_questions))
                    _eq = eval_questions[:_eval_n]
                    _eg = eval_gold_data[:_eval_n]
                    eval_metrics = self._evaluate_heldout(
                        _eq, _eg, seeker_logits_processor,
                        eval_batch_size=self.config.mini_batch_size,
                    )
                    eval_log_str = (
                        "\n[EVAL step={}] R_ans={:.4f} R_ret={:.4f} R_ins={:.4f} "
                        "R_total={:.4f} nonzero_R_ans={}% nonzero_R_ret={}% "
                        "nonzero_R_ins={}% n={}".format(
                            step, eval_metrics["avg_r_ans"],
                            eval_metrics["avg_r_ret"], eval_metrics["avg_r_ins"],
                            eval_metrics["avg_r_total"],
                            eval_metrics["nonzero_r_ans_pct"],
                            eval_metrics["nonzero_r_ret_pct"],
                            eval_metrics["nonzero_r_ins_pct"],
                            eval_metrics["n_samples"],
                        )
                    )
                    tqdm.write(eval_log_str)
                    _eval_metrics_path = os.path.join(self.args.output_dir, "eval_metrics.jsonl")
                    with open(_eval_metrics_path, "a") as f:
                        eval_metrics["step"] = step
                        f.write(json.dumps(eval_metrics) + "\n")
                except Exception as e:
                    logger.warning(f"Evaluation failed at step {step}: {e}")
                finally:
                    self.model.train()
                    torch.cuda.empty_cache()

            if (step + 1) % self.args.save_steps == 0:
                self.save_model(
                    os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
                )
                self.callback_handler.on_save(self.args, self.state, self.control)

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        # ---- Final save to ensure we have the completed model ----
        if self.is_local_process_zero():
            self.save_model(
                os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-final")
            )
            try:
                self.callback_handler.on_save(self.args, self.state, self.control)
            except Exception as e:
                logger.warning(f"on_save callback failed (non-fatal): {e}")

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _extract_question_from_text(self, text: str) -> str:
        """Extract question from batch text."""
        match = re.search(r"Question[:\s]+(.*?)(?:\n|Page|Document|$)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Fallback: return first line or full text
        return text.split("\n")[0].strip()

    def _extract_doc_page_from_path(self, image_path: str) -> Tuple[str, int]:
        """Extract document ID and 1-indexed page number from a candidate image path.

        Image paths look like ``.../img/{doc_id}_{page_num}.jpg`` where
        ``page_num`` is 1-indexed (matching the ``reference_page`` field in
        ``rag_dataset.json``).
        """
        match = re.search(r'/([^/]+)_(\d+)\.jpg$', image_path)
        if match:
            return match.group(1), int(match.group(2))
        return "", -1

    def _load_gold_answers(self, path: str) -> Dict[str, Any]:
        """Load gold answers dictionary for reward computation."""
        if not os.path.exists(path):
            logger.warning(f"Gold answers file not found at {path}. Rewards will be zero.")
            return {}
        try:
            with open(path, "r") as f:
                data = json.load(f)
            # Support both list and dict formats
            if isinstance(data, list):
                result = {}
                for item in data:
                    q = item.get("question", "").strip()
                    result[q] = item
                return result
            return data
        except Exception as e:
            logger.warning(f"Failed to load gold answers: {e}")
            return {}

    def _compute_mappo_rewards_batch(
        self,
        seeker_outputs: List[str],
        inspector_outputs: List[str],
        synthesizer_outputs: List[str],
        gold_answers: List[str],
        gold_pages: List[List[int]],
        selected_pages: List[List[int]],
        step: int = -1,
        is_eval: bool = False,
    ) -> Tuple[List[float], List[float], List[float], List[Dict[str, float]]]:
        """Compute per-agent distinct rewards using compose_mappo_rewards.

        This replaces compute_batch_rewards to match the last successful MAPPO
        run's behavior.  Each agent gets its own reward computed from its own
        action; no broadcasting.

        Returns:
            (seeker_rewards, inspector_rewards, synthesizer_rewards, reward_details)
            where reward_details is a list of per-episode dicts for logging.
        """
        seeker_rewards: List[float] = []
        inspector_rewards: List[float] = []
        synthesizer_rewards: List[float] = []
        reward_details: List[Dict[str, float]] = []

        for i in range(len(seeker_outputs)):
            gold_p = gold_pages[i] if i < len(gold_pages) else []
            sel_p = selected_pages[i] if i < len(selected_pages) else []
            gold_ans = gold_answers[i] if i < len(gold_answers) else ""

            # --- Seeker signal ---
            seeker_invalid_json = False  # seeker uses constrained decoding

            # --- Inspector signal ---
            ins_text = inspector_outputs[i].strip()
            ins_parsed = _mappo_parse_json(ins_text) if _MAPPO_RESTORE_AVAILABLE else None

            # Determine inspector_pred_sufficient
            if ins_parsed is not None:
                # Use the explicit "sufficient" field when present; a JSON like
                # {"sufficient": false, "answer": null} must NOT be classified
                # as sufficient just because the "answer" key exists.
                if "sufficient" in ins_parsed:
                    inspector_pred_sufficient = bool(ins_parsed["sufficient"])
                else:
                    inspector_pred_sufficient = bool(ins_parsed.get("answer"))
                raw_answer = ins_parsed.get("answer")
                ins_answer = "" if raw_answer is None else str(raw_answer)
            else:
                # Plain text: check for SUFFICIENT / NEED_MORE keywords
                ins_upper = ins_text.upper()
                if "SUFFICIENT" in ins_upper and "NEED_MORE" not in ins_upper and "NEED MORE" not in ins_upper:
                    inspector_pred_sufficient = True
                    ins_answer = ins_text
                elif "NEED_MORE" in ins_upper or "NEED MORE" in ins_upper:
                    inspector_pred_sufficient = False
                    ins_answer = ""
                else:
                    # Default: insufficient
                    inspector_pred_sufficient = False
                    ins_answer = ""

            # Derive gold_sufficient from whether gold pages overlap with selected pages.
            # An empty gold set means the mapping failed / evidence unavailable —
            # do NOT convert that into a positive sufficiency label.
            gold_set = set(gold_p) if gold_p else set()
            sel_set = set(sel_p) if sel_p else set()
            inspector_gold_sufficient = len(gold_set & sel_set) > 0

            # Check if inspector's answer is correct (only if inspector committed to an answer)
            if inspector_pred_sufficient and ins_answer:
                ins_f1 = _mappo_compute_f1(ins_answer, gold_ans) if _MAPPO_RESTORE_AVAILABLE else 0.0
                inspector_answer_correct = ins_f1 > 0.5
            else:
                inspector_answer_correct = None  # not applicable

            inspector_wrong_reference = False  # inspector doesn't reference pages

            # --- Synthesizer signal ---
            synth_text = synthesizer_outputs[i].strip()
            synth_parsed = _mappo_parse_json(synth_text) if _MAPPO_RESTORE_AVAILABLE else None

            if synth_parsed is not None and "answer" in synth_parsed:
                synth_answer = str(synth_parsed["answer"])
                synth_invalid_json = False
            else:
                # Plain text fallback: use raw text as answer
                synth_answer = synth_text
                synth_invalid_json = False  # treat plain text as valid

            answer_f1 = _mappo_compute_f1(synth_answer, gold_ans) if _MAPPO_RESTORE_AVAILABLE else 0.0

            # --- Compute reference_f1, groundedness, hallucinated_reference ---
            # (Matches the original _compute_batch_rewards_mappo_independent logic
            #  in vidorag_rewards_restore.py lines 1293-1311, so R_ans matches
            #  the last MAPPO run instead of being systematically lower.)
            reference_f1 = 0.0
            hallucinated_reference = False

            # Extract synth reference list from parsed JSON (if available)
            synth_ref = []
            if synth_parsed is not None and isinstance(synth_parsed, dict):
                raw_ref = synth_parsed.get("reference", [])
                if isinstance(raw_ref, list):
                    synth_ref = raw_ref
                elif raw_ref is not None:
                    synth_ref = [raw_ref]

            if isinstance(synth_ref, list) and gold_p:
                try:
                    ref_set = set(int(x) for x in synth_ref if str(x).strip().isdigit())
                    gold_set = set(gold_p)
                    if ref_set and gold_set:
                        overlap = ref_set & gold_set
                        if overlap:
                            precision = len(overlap) / len(ref_set)
                            recall = len(overlap) / len(gold_set)
                            reference_f1 = 2 * precision * recall / (precision + recall)
                    if ref_set and not (ref_set & gold_set):
                        hallucinated_reference = True
                except (ValueError, TypeError):
                    pass

            # Groundedness: simple proxy — answer F1 > 0 means some grounding.
            # (Same as the original implementation: groundedness = min(1.0, answer_f1))
            groundedness = min(1.0, answer_f1)

            # FIX B (v3): word-overlap partial credit to break bimodal R_ans.
            # answer_f1 is bimodal (71% zeros, 25% perfect 1.0), which causes
            # zero GRPO advantages when all 4 group samples get the same F1.
            # The previous length_bonus (0.15 * length_ratio) biased the
            # synthesizer toward shorter answers (avg resp len dropped from
            # 25→19 chars, hurting eval). Word overlap is semantically
            # meaningful: rewards answers containing gold answer words,
            # without any length bias.
            _gold_words = set(gold_ans.lower().split()) if gold_ans else set()
            _ans_words = set(synth_answer.lower().split()) if synth_answer else set()
            if _gold_words:
                overlap_bonus = 0.15 * (len(_gold_words & _ans_words) / len(_gold_words))
            else:
                overlap_bonus = 0.0

            turn_cost = 0.0

            # --- Compose per-agent rewards ---
            if _MAPPO_RESTORE_AVAILABLE:
                rewards_dict = compose_mappo_rewards(
                    gold_pages=gold_p,
                    selected_pages=sel_p,
                    seeker_invalid_json=seeker_invalid_json,
                    inspector_pred_sufficient=inspector_pred_sufficient,
                    inspector_gold_sufficient=inspector_gold_sufficient,
                    inspector_answer_correct=inspector_answer_correct,
                    inspector_wrong_reference=inspector_wrong_reference,
                    answer_f1=answer_f1,
                    reference_f1=reference_f1,
                    groundedness=groundedness,
                    hallucinated_reference=hallucinated_reference,
                    synth_invalid_json=synth_invalid_json,
                    overlap_bonus=overlap_bonus,
                    turn_cost=turn_cost,
                )
            else:
                # Fallback: simple rewards
                rewards_dict = {
                    "r_ret": 0.0,
                    "r_ins": 0.0,
                    "r_ans": answer_f1,
                    "r_total": answer_f1,
                    "coop_bonus": 0.0,
                }

            # --- Joint e2e override (GRPO_JOINT_E2E=1) ---
            # Recomputes the seeker and synthesizer rewards from the SAME
            # end-to-end objective, with per-agent weights. ``answer_f1`` above
            # comes from the live synthesizer rollout, so the seeker is scored
            # against the partner it is actually training with -- not against
            # the separate frozen instance that ``compute_seeker_e2e_reward``
            # would spin up via ``_get_synthesizer_model()``, which would both
            # double the forward cost and go stale as the synthesizer moves.
            if _JOINT_E2E:
                _ev = _evidence_reward_from_indices(sel_p, gold_p)
                rewards_dict["r_ret"] = max(
                    -1.0, min(1.0, _JOINT_W_EV * _ev + _JOINT_W_ANS * answer_f1)
                )
                # Synthesizer: answer side only. overlap_bonus is retained
                # because answer_f1 is bimodal (~78% exact zeros) and without a
                # partial-credit term every rollout in a group scores 0, giving
                # zero advantage and no gradient at all.
                _r_ans = answer_f1 + (overlap_bonus if _JOINT_SYNTH_OVERLAP else 0.0)
                rewards_dict["r_ans"] = max(-1.0, min(1.0, _r_ans))
                rewards_dict["r_total"] = rewards_dict["r_ret"] + rewards_dict["r_ans"]
                rewards_dict["evidence"] = _ev

            seeker_rewards.append(rewards_dict["r_ret"])
            inspector_rewards.append(rewards_dict["r_ins"])
            synthesizer_rewards.append(rewards_dict["r_ans"])
            reward_details.append(rewards_dict)

            # --- Per-episode [MAPPO-Indep] logging ---
            log_prefix = "[MAPPO-Indep-Eval]" if is_eval else "[MAPPO-Indep]"
            print(
                f"{log_prefix} step={step} episode={i} "
                f"R_ret={rewards_dict['r_ret']:.4f} "
                f"R_ins={rewards_dict['r_ins']:.4f} "
                f"R_ans={rewards_dict['r_ans']:.4f} "
                f"CoopBonus={rewards_dict['coop_bonus']:.4f} "
                f"R_total={rewards_dict['r_total']:.4f} "
                f"pred_suff={inspector_pred_sufficient} "
                f"gold_suff={inspector_gold_sufficient} "
                f"ans_f1={answer_f1:.4f} "
                f"ref_f1={reference_f1:.4f} "
                f"ground={groundedness:.4f} "
                f"halluc={hallucinated_reference} "
                f"gold_idx={gold_p} sel_idx={sel_p}",
                flush=True,
            )

        return seeker_rewards, inspector_rewards, synthesizer_rewards, reward_details

    def _log_step_rewards(
        self,
        step: int,
        seeker_reward: torch.Tensor,
        inspector_reward: torch.Tensor,
        synthesizer_reward: torch.Tensor,
    ) -> None:
        """Log per-step rewards to files."""
        os.makedirs(self.args.output_dir, exist_ok=True)
        for name, reward in [
            ("reward_seeker", seeker_reward),
            ("reward_inspector", inspector_reward),
            ("reward_synthesizer", synthesizer_reward),
        ]:
            path = os.path.join(self.args.output_dir, f"{name}.txt")
            with open(path, "a") as f:
                f.write(f"{reward.mean().item():.4f}\n")

    def _evaluate_heldout(
        self,
        eval_questions: List[str],
        eval_gold_data: List[Dict[str, Any]],
        seeker_logits_processor: LogitsProcessorList,
        eval_batch_size: int = 4,
    ) -> Dict[str, float]:
        """Run evaluation on held-out questions without PPO updates.

        Executes the full seeker -> inspector -> synthesizer pipeline and
        computes V-MMOA rewards (R_ans, R_ret, R_ins) on held-out data.
        Returns a dict of aggregated metrics.
        """
        self.model.eval()
        self.tokenizer.padding_side = "right"
        all_r_ans: List[float] = []
        all_r_ret: List[float] = []
        all_r_ins: List[float] = []
        all_r_total: List[float] = []
        eval_detail_path = os.path.join(self.args.output_dir, "eval_detail.txt")

        with torch.no_grad():
            for idx in range(0, len(eval_questions), eval_batch_size):
                batch_qs = eval_questions[idx: idx + eval_batch_size]
                batch_gd = eval_gold_data[idx: idx + eval_batch_size]
                batch_ga = [d.get("answer", "") for d in batch_gd]
                batch_gdoc = [d.get("gold_doc", "") for d in batch_gd]
                batch_gp = [d.get("reference_pages", d.get("gold_pages", [])) for d in batch_gd]

                candidate_pages = self.fetch_candidate_pages(batch_qs, top_n=10)

                batch_gold_pages: List[List[int]] = []
                for i in range(len(batch_qs)):
                    gd = batch_gdoc[i]
                    gd_stripped = os.path.splitext(gd)[0] if gd else ""
                    rs = set(batch_gp[i])
                    gi: List[int] = []
                    for ci, ip in enumerate(candidate_pages[i]):
                        did, pn = self._extract_doc_page_from_path(ip)
                        if (did == gd or did == gd_stripped) and pn in rs:
                            gi.append(ci)
                    batch_gold_pages.append(gi)

                # ---- Seeker ----
                s_msgs, s_imgs = [], []
                for i, q in enumerate(batch_qs):
                    m, im = self.get_seeker_messages(q, candidate_pages[i])
                    s_msgs.append(m)
                    s_imgs.append(im)
                s_batch = self.trans_vlm_messages_to_inputs(s_msgs, s_imgs)
                _, s_resps = self.get_vlm_inputs(s_batch, logits_processor=seeker_logits_processor, max_new_tokens=50)
                s_texts: List[str] = []
                batch_sel: List[List[int]] = []
                for i, r in enumerate(s_resps):
                    t = self.tokenizer.decode(r, skip_special_tokens=True)
                    s_texts.append(t)
                    batch_sel.append(self.parse_page_selection(t, len(candidate_pages[i])))

                # ---- Inspector ----
                i_msgs, i_imgs = [], []
                for i, q in enumerate(batch_qs):
                    si = [candidate_pages[i][k] for k in batch_sel[i] if k < len(candidate_pages[i])]
                    if not si:
                        si = candidate_pages[i][:1] if candidate_pages[i] else []
                    m, im = self.get_inspector_messages(q, si)
                    i_msgs.append(m)
                    i_imgs.append(im)
                i_batch = self.trans_vlm_messages_to_inputs(i_msgs, i_imgs)
                _, i_resps = self.get_vlm_inputs(i_batch, max_new_tokens=20)
                i_texts = [self.tokenizer.decode(r, skip_special_tokens=True) for r in i_resps]

                # ---- Synthesizer ----
                y_msgs, y_imgs = [], []
                for i, q in enumerate(batch_qs):
                    ei = [candidate_pages[i][k] for k in batch_sel[i] if k < len(candidate_pages[i])]
                    if not ei:
                        ei = candidate_pages[i][:1] if candidate_pages[i] else []
                    m, im = self.get_synthesizer_messages(q, ei)
                    y_msgs.append(m)
                    y_imgs.append(im)
                y_batch = self.trans_vlm_messages_to_inputs(y_msgs, y_imgs)
                _, y_resps = self.get_vlm_inputs(y_batch, max_new_tokens=150)
                y_texts = [self.tokenizer.decode(r, skip_special_tokens=True) for r in y_resps]

                # ---- Rewards ----
                # Use compose_mappo_rewards (per-agent distinct rewards) to match
                # the last successful MAPPO run's behavior.
                s_rw, i_rw, y_rw, _details = self._compute_mappo_rewards_batch(
                    seeker_outputs=s_texts,
                    inspector_outputs=i_texts,
                    synthesizer_outputs=y_texts,
                    gold_answers=batch_ga,
                    gold_pages=batch_gold_pages,
                    selected_pages=batch_sel,
                    step=-1,
                    is_eval=True,
                )
                s_fmt = [self.get_seeker_format_reward(t, len(candidate_pages[i])) for i, t in enumerate(s_texts)]
                s_tot = [s + f for s, f in zip(s_rw, s_fmt)]

                all_r_ret.extend(s_rw)
                all_r_ins.extend(i_rw)
                all_r_ans.extend(y_rw)
                all_r_total.extend(s_tot)

                with open(eval_detail_path, "a") as f:
                    for i in range(len(batch_qs)):
                        f.write(
                            f"[EVAL] Q={batch_qs[i][:60]}... "
                            f"R_ret={s_rw[i]:.4f} R_ins={i_rw[i]:.4f} "
                            f"R_ans={y_rw[i]:.4f} gold_idx={batch_gold_pages[i]} "
                            f"seeker={repr(s_texts[i][:80])}\n"
                        )

        self.tokenizer.padding_side = "left"
        n = len(all_r_ans)
        if n == 0:
            return {"avg_r_ans": 0.0, "avg_r_ret": 0.0, "avg_r_ins": 0.0, "avg_r_total": 0.0,
                    "nonzero_r_ans_pct": 0.0, "nonzero_r_ret_pct": 0.0, "nonzero_r_ins_pct": 0.0,
                    "n_samples": 0}
        return {
            "avg_r_ans": round(sum(all_r_ans) / n, 4),
            "avg_r_ret": round(sum(all_r_ret) / n, 4),
            "avg_r_ins": round(sum(all_r_ins) / n, 4),
            "avg_r_total": round(sum(all_r_total) / n, 4),
            "nonzero_r_ans_pct": round(sum(1 for r in all_r_ans if r > 0) / n * 100, 1),
            "nonzero_r_ret_pct": round(sum(1 for r in all_r_ret if r > 0) / n * 100, 1),
            "nonzero_r_ins_pct": round(sum(1 for r in all_r_ins if r > 0) / n * 100, 1),
            "n_samples": n,
        }

    # -----------------------------------------------------------------------
    # Overrides from parent
    # -----------------------------------------------------------------------

    @override
    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimizer(model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    @override
    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @override
    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: "torch.Tensor",
        responses: "torch.Tensor",
        model_inputs: Dict[str, Any],
        return_logits: bool = False,
        response_masks: Optional["torch.Tensor"] = None,
    ) -> Tuple["torch.Tensor", Optional["torch.Tensor"], "torch.Tensor", "torch.Tensor"]:
        """
        Calculates model outputs in multiple batches.
        Handles vision inputs (pixel_values, image_grid_thw) if present.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs: (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs: (i + 1) * fbs]
            response_batch = responses[i * fbs: (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs: (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            with self.amp_context:
                logits, _, values = model(**input_kwargs, return_dict=True, use_cache=False)

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat(
                        (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                    )[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    @override
    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Saves model checkpoint (LoRA adapter + value head)."""
        if output_dir is None:
            output_dir = self.args.output_dir

        if not self.args.should_save:
            return

        os.makedirs(output_dir, exist_ok=True)

        if self.is_fsdp_enabled or self.is_deepspeed_enabled:
            try:
                state_dict = self.accelerator.get_state_dict(self.model)
                if self.args.should_save:
                    self._save(output_dir, state_dict=state_dict)
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                if self.args.should_save:
                    self._save(output_dir, state_dict={})
                remove_dummy_checkpoint(self.args.should_save, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)
        else:
            unwrapped_model = self.accelerator.unwrap_model(self.model)

            # Save LoRA adapter via PEFT's save_pretrained
            if hasattr(unwrapped_model, 'pretrained_model'):
                backbone = unwrapped_model.pretrained_model
                if hasattr(backbone, 'save_pretrained'):
                    logger.info(f"Saving LoRA adapter to {output_dir}")
                    backbone.save_pretrained(output_dir)

                    # MAPPO independent: save all 3 adapters separately
                    # CRITICAL: Must use selected_adapters=[adapter_name] to save ONLY the
                    # specific adapter. Without it, PEFT saves ALL adapters to the top-level
                    # adapter_model.safetensors, making all adapter_* dirs identical.
                    if self._mappo_independent and hasattr(backbone, 'peft_config'):
                        for adapter_name in self._adapter_names:
                            if adapter_name == "default":
                                continue  # Already saved above
                            try:
                                backbone.set_adapter(adapter_name)
                                adapter_dir = os.path.join(output_dir, f"adapter_{adapter_name}")
                                os.makedirs(adapter_dir, exist_ok=True)
                                # Use selected_adapters to save ONLY this adapter
                                try:
                                    backbone.save_pretrained(adapter_dir, selected_adapters=[adapter_name])
                                except TypeError:
                                    # Older PEFT versions don't support selected_adapters
                                    # Fallback: save to temp dir, then copy the adapter file
                                    import tempfile, shutil
                                    with tempfile.TemporaryDirectory() as tmpdir:
                                        backbone.save_pretrained(tmpdir)
                                        for fname in os.listdir(tmpdir):
                                            shutil.copy2(os.path.join(tmpdir, fname), os.path.join(adapter_dir, fname))
                                logger.info(f"MAPPO independent: Saved {adapter_name} adapter to {adapter_dir}")
                            except Exception as e:
                                logger.warning(f"Failed to save {adapter_name} adapter: {e}")
                        # Reset to seeker adapter
                        if self._adapter_names:
                            backbone.set_adapter(self._adapter_names[0])
                else:
                    logger.warning("backbone has no save_pretrained, falling back to state_dict save")
                    import safetensors.torch
                    safetensors.torch.save_file(
                        unwrapped_model.state_dict(),
                        os.path.join(output_dir, "model.safetensors"),
                        metadata={"format": "pt"}
                    )
            else:
                import safetensors.torch
                safetensors.torch.save_file(
                    unwrapped_model.state_dict(),
                    os.path.join(output_dir, "model.safetensors"),
                    metadata={"format": "pt"}
                )

            # Save value head separately
            if hasattr(unwrapped_model, 'v_head'):
                v_head_state = {k: v.cpu() for k, v in unwrapped_model.v_head.state_dict().items()}
                import safetensors.torch
                safetensors.torch.save_file(
                    v_head_state,
                    os.path.join(output_dir, "value_head.safetensors"),
                    metadata={"format": "pt"}
                )
                logger.info(f"Saved value head to {output_dir}/value_head.safetensors")

                # GAP 3: Save per-agent value heads (3 separate files)
                if self._mappo_independent and hasattr(self, '_per_agent_v_heads'):
                    _agent_names_save = ["seeker", "inspector", "synthesizer"]
                    for _a_idx, _a_name in enumerate(_agent_names_save):
                        _vh_sd = self._per_agent_v_heads.get(_a_idx)
                        if _vh_sd is not None:
                            _vh_cpu = {k: v.cpu() if hasattr(v, 'cpu') else v
                                       for k, v in _vh_sd.items()}
                            safetensors.torch.save_file(
                                _vh_cpu,
                                os.path.join(output_dir, f"value_head_{_a_name}.safetensors"),
                                metadata={"format": "pt"}
                            )
                    logger.info(f"Saved per-agent value heads to {output_dir}/value_head_{{seeker,inspector,synthesizer}}.safetensors")

            # Save tokenizer/processor
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
