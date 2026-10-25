import json
import re


def parse_tool_output(output):
    
    tool_pattern = r"<action>\{(.*?)\}</action>"
    matches = re.findall(tool_pattern, output, re.DOTALL)
    parsed_tools = []
    for match in matches:
        try:
            # tool_data = json.loads("{" + match.strip() + "}")
            tool_data = json.loads(match.strip())
            parsed_tools.append(tool_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
    return parsed_tools

def _extract_indices_from_brackets(text):
    """Extract integer indices from a bracketed list like '[0, 1, 2]'."""
    nums = re.findall(r'\d+', text)
    return [int(n) for n in nums if int(n) < 100]  # sanity check: indices < 100


def _fallback_parse_choice(select_response):
    """Fallback parsers when standard JSON extraction fails.

    Tries multiple strategies in order of reliability:
    1. "choice": [...] pattern anywhere in text
    2. "Selected indices [X, Y]" pattern
    3. "Index X ... -> RELEVANT" markers
    4. "indices [X, Y]" generic pattern (last resort)
    """
    # Strategy 1: Look for "choice": [...] anywhere in text
    choice_match = re.search(r'"choice"\s*:\s*\[([^\]]*)\]', select_response)
    if choice_match:
        indices = _extract_indices_from_brackets(choice_match.group(1))
        if indices:
            return {
                "reason": select_response[:500],
                "summary": f"Found {len(indices)} relevant page(s) at indices {indices}",
                "choice": indices,
            }

    # Strategy 2: "Selected indices [X, Y]" pattern
    selected_match = re.search(
        r'[Ss]elected indices\s*\[([^\]]*)\]', select_response
    )
    if selected_match:
        indices = _extract_indices_from_brackets(selected_match.group(1))
        if indices:
            return {
                "reason": select_response[:500],
                "summary": f"Found {len(indices)} relevant page(s) at indices {indices}",
                "choice": indices,
            }

    # Strategy 3: "Index X ... -> RELEVANT" markers
    relevant_indices = []
    for m in re.finditer(
        r'Index\s+(\d+)\s*\([^)]*\):\s*OCR:.*?->\s*RELEVANT',
        select_response, re.DOTALL
    ):
        idx = int(m.group(1))
        if idx not in relevant_indices:
            relevant_indices.append(idx)
    if relevant_indices:
        return {
            "reason": select_response[:500],
            "summary": f"Found {len(relevant_indices)} relevant page(s) at indices {relevant_indices}",
            "choice": relevant_indices,
        }

    # Strategy 3b: Broader RELEVANT pattern (without OCR prefix)
    if not relevant_indices:
        for m in re.finditer(
            r'[Ii]ndex\s+(\d+).*?->\s*RELEVANT',
            select_response, re.DOTALL
        ):
            idx = int(m.group(1))
            if idx not in relevant_indices and idx < 100:
                relevant_indices.append(idx)
    if relevant_indices:
        return {
            "reason": select_response[:500],
            "summary": f"Found {len(relevant_indices)} relevant page(s) at indices {relevant_indices}",
            "choice": relevant_indices,
        }

    # Strategy 4: Generic "indices [X, Y]" pattern (last resort)
    indices_match = re.search(
        r'(?:at )?indices\s*\[([^\]]*)\]', select_response
    )
    if indices_match:
        indices = _extract_indices_from_brackets(indices_match.group(1))
        if indices:
            return {
                "reason": select_response[:500],
                "summary": f"Found {len(indices)} relevant page(s) at indices {indices}",
                "choice": indices,
            }

    # All strategies failed
    raise json.JSONDecodeError(
        "No valid choice or indices found in response", select_response, 0
    )


def extract_json(select_response):
    # Remove markdown code blocks
    select_response = select_response.replace('```json', '').replace('```', '')
    # Strip leading/trailing whitespace
    select_response = select_response.strip()

    # Try to find JSON object
    start_index = select_response.find('{')
    end_index = select_response.rfind('}')
    if start_index != -1 and end_index != -1 and start_index < end_index:
        json_str = select_response[start_index:end_index + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common issues: trailing commas, single quotes
            json_str_fixed = re.sub(r',\s*}', '}', json_str)
            json_str_fixed = re.sub(r',\s*]', ']', json_str_fixed)
            try:
                return json.loads(json_str_fixed)
            except json.JSONDecodeError:
                pass  # Fall through to fallback strategies

    # Fallback: try parsing entire response as JSON
    try:
        return json.loads(select_response)
    except (json.JSONDecodeError, Exception):
        pass

    # Fallback strategies for non-JSON model outputs
    return _fallback_parse_choice(select_response)


def extract_answer_from_text(text):
    """Extract a clean answer string from verbose LLM text when JSON parsing fails.

    Used by the Synthesizer as a last-resort fallback to salvage answers
    from raw verbose model outputs (e.g. "Based on the reference page
    images, the answer is 6.").

    Tries strategies in order of precision:
      1. JSON fragment with quoted "answer" field: "answer": "X"
      2. Markdown bold label: **Answer:** X or **Final Answer:** X
         (handles both same-line and next-line answer formats)
      3. Plain label at line start: Answer: X or Final Answer: X
      4. Natural language: the answer is X / supports the answer: X

    Args:
        text: Raw LLM response string that failed JSON parsing.

    Returns:
        Extracted answer string, or None if no pattern matched.
    """
    if not text or not isinstance(text, str):
        return None

    def _clean(ans):
        """Strip trailing punctuation and markdown bold wrappers."""
        ans = ans.strip().rstrip('.;').strip()
        # Strip markdown bold markers if the answer is entirely wrapped in **...**
        if ans.startswith('**') and ans.endswith('**') and len(ans) > 4:
            ans = ans[2:-2].strip()
        return ans

    # Strategy 1: JSON fragment with a quoted "answer" field.
    # Handles partial JSON like: Some text {"answer": "2002"} more text
    m = re.search(r'"answer"\s*:\s*"([^"]+)"', text)
    if m:
        ans = m.group(1).strip()
        if ans:
            return ans

    # Strategy 2: Markdown bold label: **Answer:** X, **Final Answer:** X,
    # or **Synthesized Answer:** X.
    # Handles both same-line ("**Answer:** 3") and next-line
    # ("**Answer:**\n3\n") answer formats.
    m = re.search(
        r'\*{2}(?:final\s+|synthesized\s+)?answer\s*[:：]\s*\*{0,2}\s*\n?\s*([^\n]+)',
        text, re.IGNORECASE,
    )
    if m:
        ans = _clean(m.group(1))
        if ans:
            return ans

    # Strategy 3: Plain label at line start: Answer: X, Final Answer: X,
    # or Synthesized Answer: X.
    # Requires line-start to avoid matching "answer" mid-sentence.
    m = re.search(
        r'(?:^|\n)\s*(?:final\s+|synthesized\s+)?answer\s*[:：]\s*([^\n]+)',
        text, re.IGNORECASE,
    )
    if m:
        ans = _clean(m.group(1))
        if ans:
            return ans

    # Strategy 4: Natural language phrases:
    #   - the answer is X / the final answer is X
    #   - supports the answer: X / supports the following answer: X
    #   - here is the answer: X / here is the synthesized answer: X
    m = re.search(
        r'(?:the\s+(?:final\s+)?answer\s+is'
        r'|supports\s+the\s+(?:following\s+)?answer'
        r'|here\s+is\s+the\s+(?:synthesized\s+)?answer)'
        r'\s*[:：]?\s*([^\n]+)',
        text, re.IGNORECASE,
    )
    if m:
        ans = _clean(m.group(1))
        if ans:
            return ans

    return None