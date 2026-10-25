#!/usr/bin/env python3
"""Inspector v2: prompt, retrieval cues and decision parsing.

Single source of truth for GRPO training (trainer_vidorag_inspector_grpo_v2.py),
offline checkpoint selection (eval_inspector_v2_offline.py) and closed-loop
inference (run_closedloop_eval.py, CONTROLLER_MODE=inspector_v2).

The Inspector sees the question, the Seeker's round-1 pages (at most
MAX_IMAGES) and two cues computed from the ColQwen2.5 ranking, then picks one
recovery action:

    STOP    answer from the selected pages
    DECK    complete the evidence from other pages of the same deck(s)
            (same-deck second hop, merged by union)
    EXPAND  re-select from the retrieval top 20 (merged by union)

Images are resized exactly like the GRPO trainer does (longest side <=
IMAGE_RESOLUTION, nearest neighbour), so training and inference see the same
pixels.
"""
import hashlib
import json
import os

from PIL import Image

from build_controller_sft_data_v3 import build_confidence_block

ALL_ACTIONS = ('STOP', 'DECK', 'EXPAND')
MAX_IMAGES = 8
IMAGE_RESOLUTION = 512
CONFIDENCE_DEPTH = 20     # the confidence block prints absent pages as '>20'
DECK_CONTEXT_DEPTH = 100

SYSTEM = (
    "You are an evidence auditor for question answering over slide decks. You "
    "decide whether the selected slides are sufficient to answer a question and, "
    "if not, where the missing evidence should be looked for."
)

_ACTION_LINES = {
    'STOP': "- STOP: these slides contain all the evidence needed to answer the "
            "question completely.",
    'DECK': "- DECK: part of the evidence is missing and is likely on other slides "
            "of the same deck (for example one part of a multi-part question, or a "
            "value that is defined on another slide).",
    'EXPAND': "- EXPAND: the slides look off-topic or come from the wrong deck; "
              "other highly ranked slides should be considered.",
}

TEMPLATE = (
    "Question: {question}\n\n"
    "The page selector chose the following {n} slide image(s) as evidence:\n"
    "{image_tokens}\n\n"
    "{confidence}\n\n"
    "{deck_context}\n\n"
    "Decide what to do before the answer is written:\n"
    "{action_lines}\n"
    "Trust only what is visible in the slides, never your background knowledge.\n\n"
    "Return ONLY a JSON object: {choices}."
)


def parse_actions(spec):
    """'STOP,DECK' -> ('STOP', 'DECK'); STOP must be included."""
    actions = tuple(a.strip().upper() for a in spec.split(',') if a.strip())
    unknown = [a for a in actions if a not in ALL_ACTIONS]
    if unknown or 'STOP' not in actions:
        raise ValueError(f'bad Inspector v2 action set {spec!r}')
    return actions


def deck_of(page_id):
    return page_id.rsplit('_', 1)[0]


def deck_context_block(page_ids, ranking):
    """How many decks the selection spans, and how many other pages of those
    decks the retriever ranked in its top DECK_CONTEXT_DEPTH."""
    decks = []
    for pid in page_ids:
        if deck_of(pid) not in decks:
            decks.append(deck_of(pid))
    chosen = set(page_ids)
    others = [i + 1 for i, pid in enumerate((ranking or [])[:DECK_CONTEXT_DEPTH])
              if deck_of(pid) in decks and pid not in chosen]
    lines = ['[Document Context]',
             f'The selected slides come from {len(decks)} deck(s).']
    if others:
        lines.append(f'Other slides of the same deck(s) in the retrieval top-{DECK_CONTEXT_DEPTH}: '
                     f'{len(others)} (best rank {others[0]}).')
    else:
        lines.append(f'Other slides of the same deck(s) in the retrieval top-{DECK_CONTEXT_DEPTH}: none.')
    return '\n'.join(lines)


def build_instruction(question, page_ids, ranking, actions):
    """Instruction text with one '<image>' placeholder per shown page.

    `page_ids` must already be capped to MAX_IMAGES (see shown_pages)."""
    choices = ', '.join(json.dumps({'decision': a}) for a in actions[:-1])
    choices = f'{choices} or {json.dumps({"decision": actions[-1]})}' if len(actions) > 1 \
        else json.dumps({'decision': actions[0]})
    return TEMPLATE.format(
        question=question,
        n=len(page_ids),
        image_tokens=' '.join(['<image>'] * len(page_ids)),
        confidence=build_confidence_block(page_ids, (ranking or [])[:CONFIDENCE_DEPTH]),
        deck_context=deck_context_block(page_ids, ranking),
        action_lines='\n'.join(_ACTION_LINES[a] for a in actions),
        choices=choices,
    )


def shown_pages(page_ids, img_dir):
    """(page_ids, image_paths) of the existing selected pages, at most MAX_IMAGES."""
    ids, paths = [], []
    for pid in page_ids:
        path = os.path.join(img_dir, pid + '.jpg')
        if os.path.exists(path):
            ids.append(pid)
            paths.append(path)
        if len(paths) >= MAX_IMAGES:
            break
    return ids, paths


def load_image(path):
    """Same resize as the GRPO trainer's load_and_process_images."""
    img = Image.open(path).convert('RGB')
    if max(img.width, img.height) > IMAGE_RESOLUTION:
        f = IMAGE_RESOLUTION / max(img.width, img.height)
        img = img.resize((int(img.width * f), int(img.height * f)), resample=Image.NEAREST)
    if min(img.width, img.height) < 28:
        img = img.resize((max(img.width, 28), max(img.height, 28)), resample=Image.NEAREST)
    return img


def parse_decision(text, actions):
    """{"decision": ...} -> action in `actions`, else None (noise around the JSON tolerated)."""
    start, end = text.find('{'), text.rfind('}')
    if 0 <= start < end:
        try:
            obj = json.loads(text[start:end + 1])
            d = str(obj.get('decision', '')).strip().upper().replace(' ', '_')
            if d in actions:
                return d
        except Exception:
            pass
    return None


def response_text(action):
    return json.dumps({'decision': action})


def val_split_of(file_name, holdout_mod=5):
    """Deck-level split of the official val set: 'holdout' (~1/holdout_mod of
    the decks, checkpoint selection) or 'train' (GRPO)."""
    deck = file_name[:-4] if file_name.endswith('.pdf') else file_name
    h = int(hashlib.md5(deck.encode()).hexdigest(), 16)
    return 'holdout' if h % holdout_mod == 0 else 'train'


class ActionScorer:
    """Deterministic Inspector v2 decision: mean response-token log-probability
    of each valid response string under the adapter; the decision is the argmax.

    Mean (rather than sum) is required because STOP tokenizes one token shorter
    than DECK/EXPAND, and is also exactly the response objective used by GRPO.
    Inputs are built like the GRPO trainer builds them (string chat template
    with inline vision tokens, tokenizer chat template, same image resize,
    image processor on the resized PIL images).
    """

    def __init__(self, model, processor, actions, adapter_name=None, img_dir=None):
        self.model = model
        self.processor = processor
        self.actions = tuple(actions)
        self.adapter_name = adapter_name
        self.img_dir = img_dir
        tok = processor.tokenizer
        self._resp_ids = {a: tok.encode(response_text(a), add_special_tokens=False)
                          for a in self.actions}

    def build_inputs(self, question, page_ids, ranking):
        ids, paths = shown_pages(page_ids, self.img_dir)
        instruction = build_instruction(question, ids, ranking, self.actions)
        vision_tok = '<|vision_start|><|image_pad|><|vision_end|>'
        messages = [{'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': vision_tok.join(instruction.split('<image>'))}]
        text = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images = [load_image(p) for p in paths] or None
        return self.processor(text=[text], images=images, return_tensors='pt'), ids

    def score(self, question, page_ids, ranking):
        """{action: mean response-token log p(response)} for the shown pages."""
        import torch
        if self.adapter_name is not None:
            self.model.set_adapter(self.adapter_name)
        inputs, _ids = self.build_inputs(question, page_ids, ranking)
        p_len = inputs['input_ids'].shape[1]
        out = {}
        for a in self.actions:
            resp = torch.tensor([self._resp_ids[a]], dtype=inputs['input_ids'].dtype)
            fwd = {k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
            fwd['input_ids'] = torch.cat([inputs['input_ids'], resp], dim=1)
            fwd['attention_mask'] = torch.ones_like(fwd['input_ids'])
            if 'mm_token_type_ids' in fwd:
                fwd['mm_token_type_ids'] = torch.cat(
                    [fwd['mm_token_type_ids'], torch.zeros_like(resp)], dim=1)
            fwd = {k: (v.to(self.model.device) if torch.is_tensor(v) else v)
                   for k, v in fwd.items()}
            with torch.no_grad():
                logits = self.model(**fwd).logits[0, p_len - 1:-1]
            logp = torch.log_softmax(logits.float(), dim=-1)
            out[a] = float(sum(logp[i, t] for i, t in enumerate(self._resp_ids[a]))
                           / len(self._resp_ids[a]))
        return out

    def decide(self, question, page_ids, ranking):
        scores = self.score(question, page_ids, ranking)
        return max(self.actions, key=lambda a: scores[a]), scores
