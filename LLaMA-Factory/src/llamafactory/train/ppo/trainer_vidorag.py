# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is adapted from trainer_qr_s_g.py for VLM (Qwen2.5-VL) multi-modal
# MAPPO training on ViDoRAG tasks.
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

try:
    # Shared torch-free reference implementation (also unit-tested).
    from mappo_reward_utils import sufficiency_labels as _sufficiency_labels
except ImportError:  # pragma: no cover - fallback if run outside the package dir
    def _sufficiency_labels(gold_pages, selected_pages):
        return [bool(set(gp) & set(sel))
                for gp, sel in zip(gold_pages, selected_pages)]


def _whiten_rewards(rewards: "torch.Tensor", eps: float = 1e-8) -> "torch.Tensor":
    """Zero-mean / unit-std normalize a per-agent reward tensor (FIX 3).

    Applied independently to each agent's rewards so that agents with larger
    reward magnitudes do not dominate the shared PPO value head. Degenerate
    cases (n<=1 or zero variance) are returned unchanged to avoid NaNs.
    """
    if rewards is None or rewards.numel() <= 1:
        return rewards
    std = rewards.std()
    if not torch.isfinite(std) or std < eps:
        return rewards - rewards.mean()
    return (rewards - rewards.mean()) / (std + eps)


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
        HRD_ALPHA,
        HRD_BETA,
        HRD_LAMBDA,
    )
except ImportError:
    logger.warning(
        "vidorag_rewards module not found. Using placeholder reward functions. "
        "Task 4 must create vidorag_rewards.py with the expected interface."
    )
    REWARD_METHOD = "original"
    HRD_ALPHA = 0.5
    HRD_BETA = 0.3
    HRD_LAMBDA = 0.05

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
        """Build messages for Inspector agent: decide routing (answer directly or retrieve more)."""
        system_prompt = (
            "You are a document analysis inspector. Given a question and selected document pages, "
            "decide whether the evidence is SUFFICIENT to answer the question or if MORE retrieval "
            "is needed. Output exactly one of: 'SUFFICIENT' or 'NEED_MORE'."
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
            if REWARD_METHOD == "hrd":
                logger.info(
                    "  HRD coefficients: alpha(R_ret)={}, beta(R_ins)={}, lambda(C)={}".format(
                        HRD_ALPHA, HRD_BETA, HRD_LAMBDA
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
        seeker_logits_processor = LogitsProcessorList([
            SeekerConstrainedLogitsProcessor(self.tokenizer, num_candidates=10)
        ])

        # Logging
        kl_ctl_path = os.path.join(self.args.output_dir, "kl_ctl.txt")
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(kl_ctl_path, "a") as f:
            f.write(f"init_kl_coef: {self.config.init_kl_coef}, target: {self.config.target}\n\n")

        self.config.ppo_epochs = 2

        # ---- Resume from checkpoint ----
        # In MAPPO independent mode, a checkpoint has 3 per-agent subdirectories
        # (seeker/, inspector/, synthesizer/). When resuming, we load all 3 and
        # let the multi-adapter loading block below skip re-loading them.
        # In single-adapter mode, the checkpoint has a root adapter_model.safetensors.
        if _resume_ckpt and os.path.isdir(_resume_ckpt):
            import safetensors.torch as st
            logger.info(f"Resuming from checkpoint: {_resume_ckpt}")
            unwrapped = self.accelerator.unwrap_model(self.model)
            _is_mappo_resume = os.environ.get("VIDORAG_MAPPO_INDEPENDENT", "0") == "1"
            _has_agent_subdirs = all(
                os.path.isdir(os.path.join(_resume_ckpt, _n))
                for _n in ["seeker", "inspector", "synthesizer"]
            )
            if _is_mappo_resume and _has_agent_subdirs and hasattr(unwrapped, 'pretrained_model'):
                # MAPPO independent resume: load from per-agent subdirectories
                for _name in ["seeker", "inspector", "synthesizer"]:
                    _sub = os.path.join(_resume_ckpt, _name)
                    unwrapped.pretrained_model.load_adapter(_sub, _name, is_trainable=True)
                    logger.info(f"MAPPO resume: Loaded {_name} adapter from {_sub}")
                unwrapped.pretrained_model.set_adapter("seeker")
                logger.info("MAPPO resume: Active adapter set to seeker")
            else:
                # Single-adapter resume: load root adapter as "default"
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

        # ---- MAPPO independent: load 3 warm-start adapters ----
        # Each agent (seeker/inspector/synthesizer) gets its own LoRA adapter
        # loaded from its proven per-agent PPO checkpoint. The adapters are
        # switched via set_adapter() before each agent's forward pass so the
        # correct specialized weights are used for generation and gradient updates.
        _mappo_independent = os.environ.get("VIDORAG_MAPPO_INDEPENDENT", "0") == "1"
        _agent_names = ["seeker", "inspector", "synthesizer"]
        if _mappo_independent:
            unwrapped = self.accelerator.unwrap_model(self.model)
            backbone = getattr(unwrapped, "pretrained_model", None)
            if backbone is None:
                logger.warning("VIDORAG_MAPPO_INDEPENDENT=1 but model has no pretrained_model; falling back to single adapter")
            else:
                _adapter_env_map = {
                    "seeker": "VIDORAG_SEEKER_ADAPTER",
                    "inspector": "VIDORAG_INSPECTOR_ADAPTER",
                    "synthesizer": "VIDORAG_SYNTHESIZER_ADAPTER",
                }
                for _name in _agent_names:
                    _path = os.environ.get(_adapter_env_map[_name], "")
                    if not _path or not os.path.isdir(_path):
                        logger.error(
                            f"MAPPO independent: {_adapter_env_map[_name]} not set or not a directory: {_path!r}"
                        )
                        raise FileNotFoundError(f"Warm-start adapter for {_name} not found: {_path!r}")
                    # If resuming, the adapter may already be loaded (skip reload)
                    _already = _name in getattr(backbone, "peft_config", {}) or _name in getattr(backbone, "adapters", {})
                    if _already:
                        logger.info(f"MAPPO independent: {_name} adapter already loaded, skipping")
                    else:
                        backbone.load_adapter(_path, _name, is_trainable=True)
                        logger.info(f"MAPPO independent: Loaded {_name} adapter from {_path}")
                # Set seeker as the default active adapter for the first forward pass
                backbone.set_adapter("seeker")
                logger.info("MAPPO independent: Active adapter set to seeker")

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

                # ==================== SEEKER ====================
                # Switch to seeker adapter for this agent's forward pass
                if _mappo_independent:
                    self._set_active_adapter("seeker")
                # Retrieve candidate page images
                candidate_pages = self.fetch_candidate_pages(mini_batch_questions, top_n=10)

                # Map gold reference pages (1-indexed document pages) to candidate
                # indices so the reward function can compare apples-to-apples.
                mini_batch_gold_pages = []
                for i in range(len(mini_batch_questions)):
                    gold_doc = mini_batch_gold_docs[i]
                    # Strip file extension from gold_doc for matching
                    # (gold answers use 'doc.pdf' but image paths use 'doc')
                    gold_doc_stripped = os.path.splitext(gold_doc)[0] if gold_doc else ""
                    ref_pages_set = set(mini_batch_reference_pages[i])
                    candidates = candidate_pages[i]
                    gold_candidate_indices = []
                    for cand_idx, img_path in enumerate(candidates):
                        doc_id, page_num = self._extract_doc_page_from_path(img_path)
                        if (doc_id == gold_doc or doc_id == gold_doc_stripped) and page_num in ref_pages_set:
                            gold_candidate_indices.append(cand_idx)
                    mini_batch_gold_pages.append(gold_candidate_indices)

                seeker_messages_list = []
                seeker_images_list = []
                for i, question in enumerate(mini_batch_questions):
                    msgs, imgs = self.get_seeker_messages(question, candidate_pages[i])
                    seeker_messages_list.append(msgs)
                    seeker_images_list.append(imgs)

                # Tokenize and generate
                seeker_batch = self.trans_vlm_messages_to_inputs(seeker_messages_list, seeker_images_list)
                seeker_queries, seeker_responses = self.get_vlm_inputs(
                    seeker_batch, logits_processor=seeker_logits_processor, max_new_tokens=50
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

                # ==================== INSPECTOR ====================
                # Switch to inspector adapter for this agent's forward pass
                if _mappo_independent:
                    self._set_active_adapter("inspector")
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
                    inspector_batch, max_new_tokens=20
                )

                inspector_answers_text = []
                for resp in inspector_responses:
                    text = self.tokenizer.decode(resp, skip_special_tokens=True)
                    inspector_answers_text.append(text)

                queries.extend(inspector_queries)
                responses.extend(inspector_responses)

                # ==================== SYNTHESIZER ====================
                # Switch to synthesizer adapter for this agent's forward pass
                if _mappo_independent:
                    self._set_active_adapter("synthesizer")
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
                # FIX 1: derive the Inspector's ground-truth sufficiency label
                # instead of hardcoding [True]*N. Evidence is "sufficient" only
                # when at least one gold page is present among the pages the
                # Seeker selected (what the Inspector actually inspects). With the
                # old stub the Inspector was rewarded for always answering
                # "sufficient" and penalised on genuinely-insufficient cases, so
                # its reward collapsed to ~0 / went negative and (via the shared
                # adapter) poisoned the other agents.
                has_sufficient_evidence = _sufficiency_labels(
                    mini_batch_gold_pages, mini_batch_selected_pages
                )
                batch_results = compute_batch_rewards(
                    seeker_outputs=seeker_answers_text,
                    inspector_outputs=inspector_answers_text,
                    synthesizer_outputs=synthesizer_answers_text,
                    gold_answers=mini_batch_gold_answers,
                    gold_pages=mini_batch_gold_pages,
                    total_pages=[len(pages) for pages in candidate_pages],
                    has_sufficient_evidence=has_sufficient_evidence,
                )
                seeker_rewards = torch.tensor(batch_results["seeker_rewards"], dtype=torch.float).to(self.current_device)
                inspector_rewards = torch.tensor(batch_results["inspector_rewards"], dtype=torch.float).to(self.current_device)
                synthesizer_rewards = torch.tensor(batch_results["synthesizer_rewards"], dtype=torch.float).to(self.current_device)

                # Add seeker format penalties
                seeker_format_penalties = torch.tensor([
                    self.get_seeker_format_reward(t, len(candidate_pages[i]))
                    for i, t in enumerate(seeker_answers_text)
                ], dtype=torch.float).to(self.current_device)
                seeker_total_reward = seeker_rewards + seeker_format_penalties

                rewards.extend(seeker_total_reward)
                rewards.extend(inspector_rewards)
                rewards.extend(synthesizer_rewards)

                # Log raw per-mini-batch rewards (whitening applied batch-wide below)
                self._log_step_rewards(step, seeker_total_reward, inspector_rewards, synthesizer_rewards)

            # FIX 3: per-agent reward whitening across the FULL accumulated
            # batch. Applied OUTSIDE the mini-batch loop so each agent's
            # rewards are normalized using batch-wide statistics (not the
            # per-mini-batch single-element stats that made whitening inert
            # when mini_batch_size=1). The three agents have very different
            # reward scales (seeker ~0.8, synth ~0.4, inspector ~0.04). Without
            # whitening, the largest-scale agent dominates the shared value
            # head and injects high-variance advantages. Whitening each
            # agent's rewards to zero-mean/unit-std makes contributions
            # comparable. Controlled by env MAPPO_WHITEN_PER_AGENT (default 1).
            if (
                os.environ.get("MAPPO_WHITEN_PER_AGENT", "1") == "1"
                and len(rewards) >= 3
                and len(rewards) % 3 == 0
            ):
                num_samples = len(rewards) // 3
                # Extract per-agent rewards (interleaved: s, i, y, s, i, y, ...)
                seeker_r = torch.stack(rewards[0::3])
                inspector_r = torch.stack(rewards[1::3])
                synthesizer_r = torch.stack(rewards[2::3])

                seeker_r = _whiten_rewards(seeker_r)
                inspector_r = _whiten_rewards(inspector_r)
                synthesizer_r = _whiten_rewards(synthesizer_r)

                # Reassemble in the same interleaved order
                rewards = []
                for _i in range(num_samples):
                    rewards.append(seeker_r[_i])
                    rewards.append(inspector_r[_i])
                    rewards.append(synthesizer_r[_i])

                print(
                    f"[MAPPO-Whiten] batch={num_samples} "
                    f"seeker: mean={seeker_r.mean().item():.4f} std={seeker_r.std().item():.4f} | "
                    f"inspector: mean={inspector_r.mean().item():.4f} std={inspector_r.std().item():.4f} | "
                    f"synth: mean={synthesizer_r.mean().item():.4f} std={synthesizer_r.std().item():.4f}",
                    flush=True,
                )

            # ---- PPO update phase ----
            self.model.train()
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
                def _safe_stat(key, default=0.0):
                    """Safely extract a float from PPO stats (handles tensors/arrays)."""
                    try:
                        val = stats.get(key, default)
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
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
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
        # Skip if the last periodic save already captured this step to
        # avoid an identical checkpoint-N / checkpoint-final duplicate.
        if self.is_local_process_zero():
            _final_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-final")
            _last_periodic_dir = os.path.join(
                self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            )
            if os.path.isdir(_last_periodic_dir):
                logger.info(
                    f"Last step {self.state.global_step} already saved as "
                    f"{os.path.basename(_last_periodic_dir)}; skipping duplicate final save."
                )
            else:
                self.save_model(_final_dir)
                try:
                    self.callback_handler.on_save(self.args, self.state, self.control)
                except Exception as e:
                    logger.warning(f"on_save callback failed (non-fatal): {e}")

        self.callback_handler.on_train_end(self.args, self.state, self.control)

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _set_active_adapter(self, adapter_name: str) -> None:
        """Switch the active LoRA adapter for MAPPO independent-adapter training.

        Wraps ``backbone.set_adapter(adapter_name)`` so the correct specialized
        adapter weights are used for each agent's forward pass. Also handles
        unwrapping the accelerated model.
        """
        try:
            unwrapped = self.accelerator.unwrap_model(self.model)
            backbone = getattr(unwrapped, "pretrained_model", unwrapped)
            backbone.set_adapter(adapter_name)
        except Exception as e:
            logger.warning(f"Failed to set adapter '{adapter_name}': {e}")

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
        computes HRD rewards (R_ans, R_ret, R_ins) on held-out data.
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
                br = compute_batch_rewards(
                    seeker_outputs=s_texts,
                    inspector_outputs=i_texts,
                    synthesizer_outputs=y_texts,
                    gold_answers=batch_ga,
                    gold_pages=batch_gold_pages,
                    total_pages=[len(p) for p in candidate_pages],
                    has_sufficient_evidence=_sufficiency_labels(
                        batch_gold_pages, batch_sel
                    ),
                )
                s_rw = br["seeker_rewards"]
                i_rw = br["inspector_rewards"]
                y_rw = br["synthesizer_rewards"]
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

            # Save LoRA adapter via PEFT's save_pretrained.
            # In MAPPO independent mode, save each agent's adapter to its own
            # subdirectory (seeker/, inspector/, synthesizer/) so they remain
            # separate and can be loaded independently for evaluation.
            if hasattr(unwrapped_model, 'pretrained_model'):
                backbone = unwrapped_model.pretrained_model
                if hasattr(backbone, 'save_pretrained'):
                    _is_mappo_indep = os.environ.get("VIDORAG_MAPPO_INDEPENDENT", "0") == "1"
                    _loaded_adapters = list(getattr(backbone, 'peft_config', {}).keys())
                    if _is_mappo_indep and len(_loaded_adapters) >= 3:
                        # MAPPO independent: save each adapter to its own subdirectory
                        for _adapter_name in ["seeker", "inspector", "synthesizer"]:
                            if _adapter_name not in _loaded_adapters:
                                continue
                            _subdir = os.path.join(output_dir, _adapter_name)
                            os.makedirs(_subdir, exist_ok=True)
                            logger.info(f"MAPPO independent: Saving {_adapter_name} adapter to {_subdir}")
                            try:
                                backbone.save_pretrained(_subdir, selected_adapters=[_adapter_name])
                            except TypeError:
                                # Older PEFT: save_pretrained saves all adapters.
                                # Workaround: set active, save, repeat.
                                backbone.set_adapter(_adapter_name)
                                backbone.save_pretrained(_subdir)
                        # Restore seeker as the active adapter
                        backbone.set_adapter("seeker")
                    else:
                        # Single-adapter mode: save only the active adapter
                        _active = getattr(backbone, 'active_adapter', 'default')
                        _adapters = [_active] if isinstance(_active, str) else list(_active)
                        logger.info(f"Saving LoRA adapter(s) {_adapters} to {output_dir}")
                        try:
                            backbone.save_pretrained(output_dir, selected_adapters=_adapters)
                        except TypeError:
                            backbone.save_pretrained(output_dir)
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

            # Save tokenizer/processor
            if self.processing_class is not None:
                self.processing_class.save_pretrained(output_dir)
