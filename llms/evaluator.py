from collections import Counter
import os
import re
import sys 
import time
# sys.path.append(".") 
from .llm import LLM


class JudgeUnavailableError(RuntimeError):
    """The judge backend could not be reached, so no score exists for this sample.

    Distinct from a low score on purpose: a judge outage must never be recorded as
    score=0 / passing=0, because that is indistinguishable from the model answering
    every question wrongly and mimics a catastrophic training regression.
    """


# Errors that will never succeed on retry -- billing, auth and permission faults.
# Spinning on these is what turned a DashScope 'Arrearage' (overdue payment) into a
# permanent eval hang plus a multi-megabyte log of identical 400s.
_PERMANENT_JUDGE_ERRORS = (
    'arrearage',
    'overdue',
    'invalid_api_key',
    'incorrect api key',
    'authentication',
    'unauthorized',
    'access denied',
    'permission',
    'account is in good standing',
)


def _is_permanent_judge_error(exc):
    text = str(exc).lower()
    return any(marker in text for marker in _PERMANENT_JUDGE_ERRORS)


# Retry budget. With the hard-coded 10 attempts / 30 s cap the evaluator gives up
# after ~3 minutes, which a transient network or DashScope outage easily outlasts
# (it aborted both overnight full-2215 runs after ~13 minutes of connection
# errors). Billing/auth faults still fail fast via _is_permanent_judge_error.
def _judge_max_attempts():
    raw = os.environ.get('VIDORAG_JUDGE_MAX_ATTEMPTS')
    try:
        return max(1, int(raw)) if raw else 10
    except ValueError:
        return 10


def _judge_retry_cap():
    raw = os.environ.get('VIDORAG_JUDGE_RETRY_CAP')
    try:
        return max(1, int(raw)) if raw else 30
    except ValueError:
        return 30

DEFAULT_SYSTEM_TEMPLATE = """System:
You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query and reference answer
- a generated answer

You may also be given a reference answer to use for reference in your evaluation.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains mistakes, you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, you should give a score between 4 and 5.

Example Response:
4.0
The generated answer has the exact same metrics as the reference answer, but it is not as concise.

User:
## User Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}
"""

# Binary judge prompt (VRAG-RL paper, Appendix A / Figure 12)
BINARY_SYSTEM_TEMPLATE = """Character Introduction
You are an expert evaluation system for a question answering chatbot.
You are given the following information:
- the query
- a generated answer
- a reference answer
Your task is to evaluate the correctness of the generated answer.
Response Format
Your response should be formatted as following: <judge>True or False</judge>
If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False.
Please note that the generated answer may contain additional information beyond the reference answer.

User Prompt:
Query:{query}
Reference Answer:{reference_answer}
Generated Answer:{generated_answer}"""

class Evaluator:
    """1-5 scale judge.

    The original judge model `qwen-vl-max` was retired by DashScope
    (HTTP 404 model_not_found, 2026-10-10), so the default moved to
    `qwen3-vl-plus`, its closest available successor. Calibration on 60
    stored answers: mean shift 0.0000, per-question agreement 96.7%, passing
    rate identical; the 2 disagreements were +/-1 flips on wrong answers.
    Override with the JUDGE_MODEL env var.
    """

    def __init__(self):
        self.model_name = os.environ.get('JUDGE_MODEL', 'qwen3-vl-plus')
        self.llm = LLM(self.model_name)
        self.system_template = DEFAULT_SYSTEM_TEMPLATE
        
    def llm_eval(self, query, reference_answer, generated_answer):
        system_prompt = self.system_template.format(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )
        try_times = _judge_max_attempts()
        last_error = None
        for attempt in range(try_times):
            try:
                judge = self.llm.generate(query=system_prompt)
                match = re.search(r'\d', judge)
                if match:
                    score =  int(match.group(0))
                    if score >= 4:
                        passing = 1
                    else:
                        passing = 0
                    return score, passing, judge
                # Reached the judge but got no digit back: retry, it may be a refusal
                # or a truncated reply rather than a broken backend.
                last_error = f"judge returned no score digit: {judge!r}"
                print(f"[Evaluator] attempt {attempt + 1}/{try_times}: {last_error}")
            except Exception as e:
                last_error = e
                if _is_permanent_judge_error(e):
                    # Retrying cannot help; fail immediately instead of looping.
                    raise JudgeUnavailableError(
                        f"judge backend permanently unavailable: {e}"
                    ) from e
                print(f"[Evaluator] attempt {attempt + 1}/{try_times} failed: {e}")
                time.sleep(min(2 ** attempt, _judge_retry_cap()))
        raise JudgeUnavailableError(
            f"judge failed after {try_times} attempts; last error: {last_error}"
        )
    
    def evaluate(self, query, reference_answer, generated_answer):
        system_prompt = self.system_template.format(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )
        score, passing, judge = self.llm_eval(query, reference_answer, generated_answer)
        result = dict(
            score=score,
            passing=passing,
            judge=judge,
            judge_model=self.model_name,
        )
        return result


class BinaryEvaluator:
    """Binary judge evaluator matching the VRAG-RL paper (Table 1 / Appendix A).

    Uses a text-only LLM (default: Qwen2.5-7B-Instruct via DashScope API) with
    the Figure 12 prompt that asks for ``<judge>True or False</judge>``.

    The returned ``eval_result`` dict has the same keys as ``Evaluator`` so that
    ``utils.overall_evaluator.eval_search`` works unchanged:

        - ``score``    -- 0 or 1  (mean across dataset == binary accuracy)
        - ``passing``  -- 0 or 1  (identical to score for binary mode)
        - ``judge``    -- raw judge response text

    This makes the resulting ``_eval.json`` ``score`` field directly comparable
    to the ``Accuracy`` column in VRAG-RL Table 1.
    """

    def __init__(self, model_name="qwen2.5-7b-instruct"):
        self.llm = LLM(model_name)
        self.system_template = BINARY_SYSTEM_TEMPLATE

    @staticmethod
    def _parse_judge(response):
        """Parse ``<judge>True/False</judge>`` from the judge response.

        Falls back to keyword search if the tags are missing.
        Returns 1 for True, 0 for False (or unparseable).
        """
        # Primary: extract from <judge>...</judge> tags
        match = re.search(r'<judge>\s*(True|False)\s*</judge>',
                          response, re.IGNORECASE)
        if match:
            return 1 if match.group(1).strip().lower() == 'true' else 0
        # Fallback: look for True/False as standalone tokens
        resp_lower = response.lower().strip()
        if 'false' in resp_lower:
            return 0
        if 'true' in resp_lower:
            return 1
        # Unparseable -> treat as incorrect (conservative)
        return 0

    def llm_eval(self, query, reference_answer, generated_answer):
        system_prompt = self.system_template.format(
            query=query,
            reference_answer=reference_answer,
            generated_answer=generated_answer
        )
        try_times = _judge_max_attempts()
        last_response = ""
        last_error = None
        for attempt in range(try_times):
            try:
                judge = self.llm.generate(query=system_prompt)
                last_response = judge
                binary = self._parse_judge(judge)
                return binary, binary, judge
            except Exception as e:
                last_error = e
                if _is_permanent_judge_error(e):
                    raise JudgeUnavailableError(
                        f"judge backend permanently unavailable: {e}"
                    ) from e
                print(f"[BinaryEvaluator] Attempt {attempt + 1} failed: {e}")
                time.sleep(min(2 ** attempt, _judge_retry_cap()))
        # Never return 0/0 here: a judge outage is not a wrong answer, and folding it
        # into accuracy silently deflates the reported metric.
        raise JudgeUnavailableError(
            f"judge failed after {try_times} attempts; last error: {last_error}; "
            f"last response: {last_response!r}"
        )

    def evaluate(self, query, reference_answer, generated_answer):
        score, passing, judge = self.llm_eval(query, reference_answer, generated_answer)
        return dict(
            score=score,
            passing=passing,
            judge=judge
        )


if __name__ == '__main__':
    query = "What is the capital of France?"
    reference_answer = "The capital of France is Paris."
    generated_answer = "Paris is the capital of France."

    print("=== 1-5 scale Evaluator (qwen-vl-max) ===")
    evaluator = Evaluator()
    score, passing, judge = evaluator.llm_eval(query, reference_answer, generated_answer)
    print(f"  score={score}  passing={passing}")

    print("=== Binary Evaluator (qwen2.5-7b-instruct, VRAG-RL Figure 12) ===")
    binary_evaluator = BinaryEvaluator()
    score, passing, judge = binary_evaluator.llm_eval(query, reference_answer, generated_answer)
    print(f"  binary={score}  judge={judge}")
