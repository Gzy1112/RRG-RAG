import torch
from PIL import Image
from pathlib import Path
import sys
import base64
import io
from io import BytesIO
import os
import logging

logger = logging.getLogger(__name__)


def _encode_image(image_path):
    if isinstance(image_path,Image.Image):
        buffered = io.BytesIO()
        image_path.save(buffered, format="JPEG")
        img_data = buffered.getvalue()
        base64_encoded = base64.b64encode(img_data).decode("utf-8")
        return base64_encoded
    else:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


def _image_item(img):
    """Qwen-VL image content entry. VIDORAG_IMAGE_MAX_PIXELS (unset = native up to
    qwen_vl_utils' 16384-token default) caps the pixels per page; high-resolution
    benchmarks (ViDoSeek/MMLongBench pages up to 10 MP) otherwise OOM the 7B agents."""
    item = dict(type="image", image=img)
    max_pixels = os.environ.get('VIDORAG_IMAGE_MAX_PIXELS')
    if max_pixels:
        item['max_pixels'] = int(max_pixels)
    return item


def _is_lora_adapter_path(path):
    """Check if a given path is a directory containing LoRA adapter_config.json."""
    p = Path(path)
    return p.is_dir() and (p / "adapter_config.json").exists()

class Qwen_VL_2_5:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto",attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def generate(self,query, images):
        from qwen_vl_utils import process_vision_info
        if not isinstance(query, str):
            raise ValueError("Query must be a string")
        content = [dict(
            type = "image",
            image = img
        ) for img in images]
        content.append(dict(type='text',text=query))
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=1028)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]


class TrainedVLM:
    """Loads a VLM with LoRA adapters for inference.
    
    Provides the same generate(query, images) interface as other VLM classes.
    Auto-detects the base model from adapter_config.json if not specified.
    
    Args:
        adapter_path: Path to the LoRA adapter directory (contains adapter_config.json).
        base_model_name: Base model name/path. If None, reads from adapter_config.json.
        merge_lora: If True, merge LoRA weights into base model for faster inference.
                    If False, keep adapters separate (more memory-efficient).
    """

    def __init__(self, adapter_path, base_model_name=None, merge_lora=False, device=None):
        import json as _json
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "The 'peft' library is required to load LoRA adapters. "
                "Install it with: pip install peft"
            )
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Auto-detect base model from adapter_config.json
        if base_model_name is None:
            adapter_config_path = Path(adapter_path) / "adapter_config.json"
            with open(adapter_config_path, "r") as f:
                adapter_config = _json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")
            if not base_model_name:
                raise ValueError(
                    f"Could not find 'base_model_name_or_path' in {adapter_config_path}. "
                    "Please specify base_model_name explicitly."
                )

        _device_map = {"": device} if device else "auto"
        logger.info(f"Loading base model: {base_model_name} (device_map={_device_map})")
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map=_device_map,
            trust_remote_code=True,
        )

        # Qwen2.5-VL defines _no_split_modules as a set, which causes
        # accelerate's get_balanced_memory to crash with "unhashable type:
        # 'set'" when loading multiple models (PEFT passes it through
        # set() internally).  Convert to list as a safety measure.
        if hasattr(base_model, '_no_split_modules') and isinstance(base_model._no_split_modules, set):
            base_model._no_split_modules = list(base_model._no_split_modules)

        logger.info(f"Loading LoRA adapters from: {adapter_path}")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)

        if merge_lora:
            logger.info("Merging LoRA weights into base model...")
            self.model = self.model.merge_and_unload()
            logger.info("LoRA merge complete.")
        else:
            self.model.eval()

        self.processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        logger.info("TrainedVLM ready for inference.")

    def generate(self, query, images, system_prompt=None):
        """Generate response given a query and list of images.
        
        Args:
            query: Text query string. May contain <image> placeholders to indicate
                   where each image should be positioned in the conversation.
            images: List of image paths (str/Path) or PIL.Image objects.
            system_prompt: Optional system prompt string.
        
        Returns:
            Generated text response.
        """
        from qwen_vl_utils import process_vision_info

        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        # Build user content with images at correct positions
        if "<image>" in query:
            # Split query around <image> placeholders and interleave with image items
            parts = query.split("<image>")
            content = []
            img_idx = 0
            for i, part in enumerate(parts):
                if part:  # Add text segment (skip empty strings)
                    content.append(dict(type="text", text=part))
                if i < len(parts) - 1 and img_idx < len(images):
                    content.append(_image_item(images[img_idx]))
                    img_idx += 1
        else:
            # Fallback: images first, then text
            content = [_image_item(img) for img in images]
            content.append(dict(type="text", text=query))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1028,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0]
    
        
class SharedMultiAdapterVLM:
    """Loads ONE base model and attaches multiple LoRA adapters to it.

    This avoids loading N separate copies of the base model when running
    multiple agents (Seeker, Inspector, Synthesizer) that each have their
    own LoRA adapter but share the same underlying VLM architecture.

    Memory savings: ~(N-1) * base_model_size  (e.g. 2 * 7 GB = 14 GB for 3 agents).

    Usage::

        group = SharedMultiAdapterVLM(
            adapter_paths={'seeker': '/path/to/seeker', 'inspector': '/path/to/insp', ...},
            base_model_name='/path/to/Qwen2.5-VL-3B',  # optional, auto-detected
        )
        seeker_llm = group.get_agent('seeker')
        seeker_llm.generate(query, images)
    """

    def __init__(self, adapter_paths: dict, base_model_name: str = None):
        import json as _json
        from peft import PeftModel, get_peft_model, LoraConfig
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Auto-detect base model from the first adapter's adapter_config.json
        if base_model_name is None:
            first_adapter = next(iter(adapter_paths.values()))
            cfg_path = Path(first_adapter) / "adapter_config.json"
            with open(cfg_path) as f:
                cfg = _json.load(f)
            base_model_name = cfg.get("base_model_name_or_path")
            if not base_model_name:
                raise ValueError(f"Cannot detect base model from {cfg_path}")

        logger.info(f"[SharedMulti] Loading base model ONCE: {base_model_name}")
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Qwen2.5-VL defines _no_split_modules as a set → crashes accelerate
        if hasattr(base_model, '_no_split_modules') and isinstance(base_model._no_split_modules, set):
            base_model._no_split_modules = list(base_model._no_split_modules)

        # Load the FIRST adapter to initialise the PeftModel wrapper,
        # then rename its adapter so the name matches our key.
        first_name, first_path = next(iter(adapter_paths.items()))
        logger.info(f"[SharedMulti] Primary adapter: {first_name} = {first_path}")
        self._peft_model = PeftModel.from_pretrained(base_model, first_path)
        # The default adapter name in PEFT is "default".  Rename it.
        if first_name != "default":
            self._peft_model._internal_dict = {}
            # Use load_adapter with the correct name, then remove the "default"
            self._peft_model.load_adapter(first_path, adapter_name=first_name)
            # Delete the original "default" adapter
            self._peft_model.delete_adapter("default")
        self._peft_model.set_adapter(first_name)
        self._peft_model.eval()

        # Load remaining adapters with explicit names
        for name, path in adapter_paths.items():
            if name == first_name:
                continue
            logger.info(f"[SharedMulti] Loading additional adapter: {name} = {path}")
            self._peft_model.load_adapter(path, adapter_name=name)

        self.processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        self._adapter_names = list(adapter_paths.keys())
        # Keep the name -> adapter directory mapping so each proxy can report the
        # same introspection attributes a standalone LLM(model_name=<adapter dir>)
        # reports (see _AgentProxy). Without it the agents cannot tell that they
        # are driving a fine-tuned VLM.
        self._adapter_paths = dict(adapter_paths)
        self._base_model_name = base_model_name
        logger.info(f"[SharedMulti] Ready. Adapters: {self._adapter_names}")

    def get_agent(self, name: str):
        """Return a lightweight _AgentProxy that shares the base model."""
        if name not in self._adapter_names:
            raise ValueError(f"Unknown agent '{name}'. Available: {self._adapter_names}")
        return _AgentProxy(self, name)

    def get_raw_agent(self):
        """Return a proxy that drives the shared base model with every adapter
        disabled, behaving exactly like LLM(model_name=<base dir>) (LocalVLM)."""
        return _RawBaseProxy(self)

    def _set_active_adapter(self, name: str):
        self._peft_model.set_adapter(name)


class _AgentProxy:
    """Lightweight proxy that switches to the correct adapter before generate().

    Introspection parity with LLM: vidorag_agents.py decides each agent's prompt
    format from ``getattr(vlm, '_is_trained_vlm', False)`` (Seeker:165,
    Inspector:322, Synthesizer:430) and, for the Seeker, from
    ``getattr(vlm, '_adapter_path', '')`` (Seeker:174, which detects the v4/v5
    choice-only format via 'v4'/'v5'/'tfr' in the adapter path). A proxy that
    omits them makes every agent silently fall back to the legacy API-model
    branch (concatenated image grid + page_map prompt), i.e. a prompt the SFT
    adapters were never trained on. These attributes restore exactly what
    ``LLM(model_name=<adapter dir>)`` exposed before the shared-base path existed,
    so the shared-base optimisation is behaviour-neutral.
    """

    def __init__(self, group: SharedMultiAdapterVLM, agent_name: str):
        self._group = group
        self._name = agent_name
        adapter_path = group._adapter_paths.get(agent_name, '')
        self._adapter_path = adapter_path
        # SharedMultiAdapterVLM only ever loads LoRA adapter directories, but
        # reuse the same predicate LLM.__init__ uses so the two stay in sync.
        self._is_trained_vlm = bool(adapter_path) and _is_lora_adapter_path(adapter_path)
        self._is_local_vlm = False
        self.model_name = adapter_path or getattr(group, '_base_model_name', '')

    @property
    def model(self):
        return self._group._peft_model

    @property
    def processor(self):
        return self._group.processor

    def generate(self, **kwargs):
        from qwen_vl_utils import process_vision_info

        query = kwargs.get('query', '')
        images = kwargs.get('image', kwargs.get('images', ''))
        system_prompt = kwargs.get('system_prompt', None)

        # Switch to this agent's adapter
        self._group._set_active_adapter(self._name)

        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        if "<image>" in query:
            parts = query.split("<image>")
            content = []
            img_idx = 0
            for i, part in enumerate(parts):
                if part:
                    content.append(dict(type="text", text=part))
                if i < len(parts) - 1 and img_idx < len(images):
                    content.append(_image_item(images[img_idx]))
                    img_idx += 1
        else:
            content = [_image_item(img) for img in images]
            content.append(dict(type="text", text=query))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})

        text = self._group.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._group.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self._group._peft_model.device)

        generated_ids = self._group._peft_model.generate(
            **inputs, max_new_tokens=1028, do_sample=False,
            temperature=None, top_p=None,
            pad_token_id=self._group.processor.tokenizer.eos_token_id,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._group.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


class _RawBaseProxy:
    """Raw base model served from a SharedMultiAdapterVLM with all LoRA adapters
    disabled, so a mixed config (some agents fine-tuned, some raw) holds ONE 7B copy
    instead of a TrainedVLM base plus a LocalVLM base (~31 GB, OOMs a 40 GB card
    next to the retrieval server).

    Introspection and generate() mirror LLM(model_name=<base dir>) -> LocalVLM:
    _is_trained_vlm False (legacy prompts, tiled page canvas), images placed before
    the text without per-page max_pixels, the same greedy decoding."""

    def __init__(self, group: SharedMultiAdapterVLM):
        self._group = group
        self._is_trained_vlm = False
        self._is_local_vlm = True
        self.model_name = group._base_model_name

    def generate(self, **kwargs):
        from qwen_vl_utils import process_vision_info

        query = kwargs.get('query', '')
        images = kwargs.get('image', '')
        system_prompt = kwargs.get('system_prompt', None)
        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        content = [dict(type="image", image=img) for img in images]
        content.append(dict(type="text", text=query))
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})

        processor = self._group.processor
        peft_model = self._group._peft_model
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(peft_model.device)

        with peft_model.disable_adapter():
            generated_ids = peft_model.generate(
                **inputs, max_new_tokens=1028, do_sample=False,
                temperature=None, top_p=None,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


class LocalVLM:
    """Loads a raw local VLM model (no LoRA) for inference.

    Supports any model loadable via AutoModelForImageTextToText (e.g. Qwen3.5).
    Provides the same generate(query, images) interface as other VLM classes.
    """

    def __init__(self, model_path):
        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info(f"Loading local VLM from: {model_path}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        logger.info("LocalVLM ready for inference.")

    def generate(self, query, images, system_prompt=None):
        """Generate response given a query and list of images.

        Args:
            query: Text query string.
            images: List of image paths (str/Path) or PIL.Image objects.
            system_prompt: Optional system prompt string.

        Returns:
            Generated text response.
        """
        from qwen_vl_utils import process_vision_info

        if not isinstance(query, str):
            raise ValueError("Query must be a string")

        content = [dict(type="image", image=img) for img in images]
        content.append(dict(type="text", text=query))

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({"role": "user", "content": content})

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1028,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


class LLM:
    def __init__(self, model_name, merge_lora=False, device=None):
        self.model_name = model_name
        self.merge_lora = merge_lora
        self._is_trained_vlm = False
        self._is_local_vlm = False

        # Check if model_name is a path to a LoRA adapter directory
        if _is_lora_adapter_path(model_name):
            logger.info(f"Detected LoRA adapter at: {model_name}")
            self._is_trained_vlm = True
            self._adapter_path = model_name
            self.model = TrainedVLM(adapter_path=model_name, merge_lora=merge_lora, device=device)
        elif os.path.isdir(model_name) and not _is_lora_adapter_path(model_name):
            # Raw local model directory (no LoRA adapters)
            logger.info(f"Detected local model directory: {model_name}")
            self._is_local_vlm = True
            self.model = LocalVLM(model_path=model_name)
        elif 'Qwen2.5-VL' in self.model_name:
            self.model = Qwen_VL_2_5(model_name)
        elif model_name.startswith('gpt'):
            from openai import OpenAI
            self.model = OpenAI()
        elif 'qwen' in self.model_name.lower():
            from openai import OpenAI
            self.model = OpenAI(
                api_key=os.environ.get("DASHSCOPE_API_KEY"),
                base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
            )
            self.model_name = model_name  # Store actual model name
            
    def generate(self,**kwargs):
        query = kwargs.get('query','')
        image = kwargs.get('image','')
        model_name = kwargs.get('model_name','')
        system_prompt = kwargs.get('system_prompt', None)

        if self._is_trained_vlm or self._is_local_vlm:
            return self.model.generate(query, image, system_prompt=system_prompt)
        elif 'Qwen2.5-VL' in self.model_name:
            return self.model.generate(query,image)
        elif self.model_name.startswith('gpt'):
            content = [{
                "type": "text",
                "text": query
            }]
            if image != '':
                filepaths = [Path(img).resolve().as_posix() for img in image]
                for filepath in filepaths:
                    base64_image = _encode_image(filepath)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"}}
                        )
            completion = self.model.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                    "role": "user",
                    "content": content
                    }
                ])
            return completion.choices[0].message.content
        elif 'qwen' in self.model_name.lower():
            # DashScope OpenAI-compatible API
            content = []
            if image != '':
                # Multi-modal: with images
                for img_path in image:
                    base64_image = _encode_image(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    })
            content.append({"type": "text", "text": query})
            
            completion = self.model.chat.completions.create(
                model=self.model_name,  # Use actual model name (e.g., qwen-vl-max)
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ]
            )
            return completion.choices[0].message.content
if __name__ == '__main__':
    llm = LLM("qwen-vl-max")  #('gpt-4o')
    response = llm.generate(query='describe in 3 words',image=['../data/MMLongBench/img/0b85477387a9d0cc33fca0f4becaa0e5_1.jpg'])
    print(response)