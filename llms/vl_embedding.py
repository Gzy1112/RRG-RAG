import asyncio
import os
from typing import Any, List, Optional, Union

import torch
import torch.nn.functional as F
from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
    ColQwen2,
    ColQwen2Processor,
)
from llama_index.core.base.embeddings.base import Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings import MultiModalEmbedding
from PIL import Image
from transformers import AutoModel, AutoTokenizer


def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps


class VL_Embedding(MultiModalEmbedding):
    model: str = Field(description="The Multi-model to use.")

    api_key: Optional[str] = Field(
        default=None,
        description="The API key.",
    )
    dimensions: Optional[int] = Field(
        default=1024,
        description=(
            "The number of dimensions the resulting output embeddings should have. "
            "Only supported in embedding-3 and later models. embedding-2 is fixed at 1024."
        ),
    )
    timeout: Optional[float] = Field(
        default=None,
        description="The timeout.",
    )

    mode: str = Field(
        default="text",
        description="The mode of the model, either 'text' or 'image'.",
    )
    show_progress: bool = Field(
        default=False,
        description="Whether to show progress bars.",
    )

    embed_model: Union[ColQwen2, AutoModel, None] = Field(default=None)
    processor: Optional[ColQwen2Processor] = Field(default=None)
    tokenizer: Optional[AutoTokenizer] = Field(default=None)

    def __init__(
        self,
        model: str = "vidore/colqwen2-v1.0",
        dimensions: Optional[int] = 1024,
        timeout: Optional[int] = None,
        callback_manager: Optional[CallbackManager] = None,
        mode: str = "text",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            dimensions=dimensions,
            timeout=timeout,
            callback_manager=callback_manager,
            **kwargs,
        )

        self.mode = mode

        if "openbmb" in model:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, trust_remote_code=True
            )
            self.embed_model = (
                AutoModel.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="cuda:1",
                )
                .cuda()
                .eval()
            )
            # self.embed_model.eval()
        elif "vidore" in model and "qwen" in model:
            self.embed_model = ColQwen2.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="cuda",  # or "mps" if on Apple Silicon
            ).eval()
            self.processor = ColQwen2Processor.from_pretrained(model)

            # Fix key mapping: checkpoint stores embed_tokens and norm under
            # 'model.*' prefix, but ColQwen2 expects 'language_model.*'.
            # Without this fix, embed_tokens (token embedding table) and
            # norm (final layer norm) are randomly initialized, making
            # all query embeddings garbage.
            self._fix_colqwen2_key_mapping(model)
        elif "vidore" in model and "pali" in model:
            self.embed_model = ColPali.from_pretrained(
                model,
                torch_dtype=torch.bfloat16,
                device_map="cuda",  # or "mps" if on Apple Silicon
            ).eval()
            self.processor = ColPaliProcessor.from_pretrained(model)

    @classmethod
    def class_name(cls) -> str:
        return "VL_Embedding"

    def _fix_colqwen2_key_mapping(self, model_path: str):
        """Fix ColQwen2 checkpoint key mapping.

        The checkpoint stores embed_tokens and norm under 'model.*' prefix,
        but ColQwen2 model expects 'language_model.*' prefix.
        HuggingFace's from_pretrained remaps most keys (model.layers.* →
        language_model.layers.*) but misses model.embed_tokens.weight and
        model.norm.weight, leaving them randomly initialized.

        This method loads those 2 missing weights from the safetensors
        checkpoint, remaps the keys, and loads them into the model.
        """
        import json as _json
        from safetensors.torch import load_file

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            # Single-file checkpoint
            st_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(st_path):
                return
            tensors = load_file(st_path)
            files_to_check = {st_path: tensors}
        else:
            index = _json.load(open(index_path))
            weight_map = index.get("weight_map", {})
            # Keys that need remapping
            keys_to_remap = [
                k for k in weight_map if k.startswith("model.")
                and k not in (
                    "model.layers",  # already remapped by HF
                )
            ]
            # Focus on the known missing keys
            priority_keys = [
                "model.embed_tokens.weight",
                "model.norm.weight",
            ]
            keys_to_remap = [k for k in priority_keys if k in weight_map]
            if not keys_to_remap:
                return
            # Find which safetensors files contain these keys
            files_to_load = set(weight_map[k] for k in keys_to_remap)
            files_to_check = {
                os.path.join(model_path, f): load_file(os.path.join(model_path, f))
                for f in files_to_load
            }

        # Remap model.* → language_model.*
        remapped = {}
        for fpath, tensors in files_to_check.items():
            for key in ["model.embed_tokens.weight", "model.norm.weight"]:
                if key in tensors:
                    new_key = "language_model." + key[len("model."):]
                    remapped[new_key] = tensors[key].to(torch.bfloat16)

        if not remapped:
            return

        # Load into model
        self.embed_model.load_state_dict(remapped, strict=False)
        print(f"[VL_Embedding] Fixed key mapping: loaded {len(remapped)} keys: "
              f"{list(remapped.keys())}")

    def embed_img(self, img_path):
        if isinstance(img_path, str):
            img_path = [img_path]
        if "vidore" in self.model:
            images = [Image.open(img) for img in img_path]
            batch_images = self.processor.process_images(images).to(
                self.embed_model.device
            )
            with torch.no_grad():
                image_embeddings = self.embed_model(**batch_images)
        elif "openbmb" in self.model:
            images = [Image.open(img).convert("RGB") for img in img_path]
            inputs = {
                "text": [""] * len(images),
                "image": images,
                "tokenizer": self.tokenizer,
            }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)
                image_embeddings = (
                    F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                )
                # image_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().tolist()[0]
            # image_embeddings = embeddings.tolist()[0]
        return image_embeddings

    def embed_text(self, text):
        if isinstance(text, str):
            text = [text]
        if "colqwen" in self.model:
            batch_queries = self.processor.process_queries(text).to(
                self.embed_model.device
            )
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif "colpali" in self.model:
            batch_queries = self.processor.process_queries(text).to(
                self.embed_model.device
            )
            with torch.no_grad():
                query_embeddings = self.embed_model(**batch_queries)
        elif "openbmb" in self.model:
            INSTRUCTION = (
                "Represent this query for retrieving relevant documents: "
            )
            queries = [INSTRUCTION + query for query in text]
            inputs = {
                "text": queries,
                "image": [None] * len(queries),
                "tokenizer": self.tokenizer,
            }
            with torch.no_grad():
                outputs = self.embed_model(**inputs)
                attention_mask = outputs.attention_mask
                hidden = outputs.last_hidden_state
                reps = weighted_mean_pooling(hidden, attention_mask)
                # query_embeddings = F.normalize(reps, p=2, dim=1).detach().cpu().numpy()
                query_embeddings = (
                    F.normalize(reps, p=2, dim=1).detach().cpu().tolist()
                )
                # query_embeddings = embeddings.tolist()[0]
        return query_embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings."""
        embeddings_list: List[List[float]] = []
        for text in texts:
            embeddings = self.embed_text(text)
            embeddings = embeddings[0]
            embeddings_list.append(embeddings)
        return embeddings_list

    def _aget_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self.embed_text(query)[0]

    def _aget_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self.embed_text(text)[0]

    def _get_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)

    def _aget_image_embedding(self, img_file_path) -> Embedding:
        return self.embed_img(img_file_path)

    def __call__(self, nodes, **kwargs):
        if "vidore" in self.model:
            if self.mode == "image":
                embeddings = self.embed_img(
                    [node.metadata["file_path"] for node in nodes]
                )
                embeddings = embeddings.view(embeddings.size(0), -1).tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                embeddings = embeddings.view(embeddings.size(0), -1).tolist()

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

        elif "openbmb" in self.model:
            if self.mode == "image":
                embeddings = self.embed_img(
                    [node.metadata["file_path"] for node in nodes]
                )
                embeddings = embeddings.tolist()
            else:
                embeddings = self.embed_text([node.text for node in nodes])
                # embeddings = embeddings.tolist()
                # embeddings = [embeddings]

            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding

        return nodes

    def score(self, image_embeddings, text_embeddings):
        if "vidore" in self.model:
            score = self.processor.score_multi_vector(
                image_embeddings, text_embeddings
            )
        elif "openbmb" in self.model:
            score = text_embeddings @ image_embeddings.T
        return score


if __name__ == "__main__":
    colpali = VL_Embedding("vidore/colqwen2-v1.0")
    image_embeddings = colpali.embed_img(
        "./data/ExampleDataset/img/00a76e3a9a36255616e2dc14a6eb5dde598b321f_1.jpg"
    )
    text_embeddings = colpali.embed_text("Hello, world!")
    score = colpali.processor.score_multi_vector(
        image_embeddings, text_embeddings
    )
    # check shapes
    print(image_embeddings.shape)  # torch.Size([1, 779, 128])
    print(text_embeddings.shape)  # torch.Size([1, 26, 128])
    print(score)  # tensor([[115.5000]])
