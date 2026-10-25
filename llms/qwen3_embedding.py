"""Qwen3-Embedding wrapper (dense, 1024-dim, last-token pooling, L2-normalized).

Drop-in replacement for the BGE-M3 text embedder used by the ViDoRAG text
retrieval path. Implemented directly on top of transformers so it does not
depend on sentence-transformers / SentenceTransformer module configs.

Asymmetric usage (per the official Qwen3-Embedding recipe):
  - documents are embedded with NO instruction prefix
  - queries are embedded with the retrieval instruction prefix
"""

import os
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from llama_index.core.base.embeddings.base import BaseEmbedding, Embedding
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from transformers import AutoModel, AutoTokenizer

DEFAULT_QWEN3_EMBED_PATH = os.environ.get(
    'QWEN3_EMBED_MODEL_PATH',
    'Qwen/Qwen3-Embedding-0.6B',
)

# Matches config_sentence_transformers.json -> prompts.query (a trailing space is
# added because the official prompt is joined directly with the query string).
QWEN3_QUERY_INSTRUCTION = (
    'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '
)


def is_qwen3_embedding(model_name: str) -> bool:
    """True for Qwen3-Embedding model ids/paths, False for ColQwen* VL models."""
    name = (model_name or '').lower()
    if 'colqwen' in name:
        return False
    return 'qwen3-embedding' in name or 'qwen3_embedding' in name or 'qwen3' in name


def last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padded = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padded:
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), lengths]


class Qwen3Embedding(BaseEmbedding):
    """LlamaIndex embedding backed by Qwen3-Embedding-0.6B."""

    max_length: int = Field(default=1024, description='Max tokens per text (includes the EOS token).')
    query_instruction: str = Field(default=QWEN3_QUERY_INSTRUCTION, description='Prefix prepended to queries only.')
    device: str = Field(default='cuda', description='Device used for the forward pass.')
    torch_dtype: str = Field(default='float16', description='Weight dtype: float16 / bfloat16 / float32.')

    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(
        self,
        model_name: str = DEFAULT_QWEN3_EMBED_PATH,
        embed_batch_size: int = 64,
        max_length: int = 1024,
        query_instruction: str = QWEN3_QUERY_INSTRUCTION,
        device: str = 'cuda',
        torch_dtype: str = 'float16',
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name,
            embed_batch_size=embed_batch_size,
            max_length=max_length,
            query_instruction=query_instruction,
            device=device,
            torch_dtype=torch_dtype,
            **kwargs,
        )
        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
            self.device = device
        dtype = getattr(torch, torch_dtype)
        # Left padding keeps the pooled last token at position -1 for every row.
        self._tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if model is not None:
            self._model = model.to(device).eval()
        elif device.startswith('cuda'):
            # Place weights directly on the target device via accelerate dispatch
            # (the pattern VL_Embedding uses for ColQwen2). A post-hoc
            # .to('cuda') that follows other accelerate device_map="auto" model
            # loads can raise "CUDA driver error: invalid argument" when
            # PYTORCH_CUDA_ALLOC_CONF=expandable_segments is set; loading straight
            # onto the device sidesteps that and the transient host->device copy.
            self._model = AutoModel.from_pretrained(model_name, dtype=dtype, device_map=device).eval()
        else:
            self._model = AutoModel.from_pretrained(model_name, dtype=dtype).to(device).eval()

    @classmethod
    def class_name(cls) -> str:
        return 'qwen3_embedding'

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> List[List[float]]:
        batch = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}
        hidden = self._model(**batch).last_hidden_state
        pooled = last_token_pool(hidden, batch['attention_mask'])
        pooled = F.normalize(pooled.float(), p=2, dim=1)
        return pooled.cpu().tolist()

    def embed_documents(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Embed document-side texts (no instruction prefix), length-sorted for speed."""
        batch_size = batch_size or self.embed_batch_size
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out: List[Optional[List[float]]] = [None] * len(texts)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            for i, vec in zip(idx, self._encode([texts[i] for i in idx])):
                out[i] = vec
        return out  # type: ignore[return-value]

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._encode([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return self._encode(list(texts))

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._encode([self.query_instruction + query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return self._get_text_embedding(text)
