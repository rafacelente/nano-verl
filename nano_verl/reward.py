import re

def extract_boxed_answer(response: str) -> str:
    idx = response.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    start = i
    while len(response) and depth > 0:
        if response[i] == "{":
            depth += 1
        elif response[i] == "}":
            depth -= 1
        i+=1
    if depth != 0:
        return None
    return response[start:(i-1)]

def normalize_answer(answer: str) -> str:
    answer = re.sub(r"\s+", "", answer)
    answer = answer.replace("$", "")
    return answer

def compute_score(response: str, ground_truth: str) -> float:
    extracted = extract_boxed_answer(response)
    if extracted is None:
        return 0.0

    cleaned_response = normalize_answer(extracted)
    cleaned_gt = normalize_answer(ground_truth)
    return 1.0 if cleaned_response == cleaned_gt else 0.0
