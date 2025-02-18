import os
import json
import enchant
import random
from dotenv import load_dotenv
load_dotenv()

import llm  # <-- new import

with open('info.json', 'r') as file:
    data = json.load(file)

instructions = data["instructions_w"]
objective = data["objective_w"]

def load_words(file_path):
    with open(file_path, 'r') as file:
        return [line.strip().lower() for line in file if len(line.strip()) == 5]

def colorize_guess(guess, target):
    """Generates G/Y/_ feedback for guess vs. target."""
    result = ['_'] * 5
    target_tmp = list(target)
    feedback_details = []

    # First pass: correct positions
    for i in range(5):
        if guess[i] == target[i]:
            result[i] = 'G'
            target_tmp[i] = None
            feedback_details.append(f"Position {i+1}: {guess[i]} - G")
        else:
            feedback_details.append(f"Position {i+1}: {guess[i]} - _")

    # Second pass: correct letter, wrong position
    for i in range(5):
        if guess[i] != target[i] and guess[i] in target_tmp:
            result[i] = 'Y'
            # Mark that letter used
            index_ = target_tmp.index(guess[i])
            target_tmp[index_] = None
            # Update feedback details line
            feedback_details[i] = f"Position {i+1}: {guess[i]} - Y"

    GYs = ''.join(result)
    detailed_feedback = "\n".join(feedback_details)
    return GYs, detailed_feedback

def check_word_validity(word):
    """Check if word is real English (and not empty)."""
    if not word:
        return False
    d = enchant.Dict("en_US")
    return d.check(word)

def extract_word(response_str):
    """
    Attempt to parse a single guess from the LLM's returned string.
    """
    try:
        # Often the LLM might wrap in JSON
        parsed = json.loads(response_str)
        if isinstance(parsed, dict):
            for key, val in parsed.items():
                if isinstance(val, str):
                    return val.strip()
        return ""
    except (json.JSONDecodeError, TypeError):
        # fallback: raw string
        return response_str.strip()

def play_wordle(file_path, run_id, llm_type, results):
    words = load_words(file_path)
    target = random.choice(words)
    attempts = 0
    max_attempts = 5
    guess_history = []

    # Map llm_type to model name:
    if llm_type == "openai":
        model_name = data.get("GPT_MODEL", "gpt-4o-mini")
    elif llm_type == "claude":
        model_name = data.get("CLAUDE", "claude-3-5-sonnet-20240620")
    elif llm_type == "groq":
        model_name = "groq-model"
    elif llm_type == "gemini":
        model_name = data.get("GEMINI", "gemini-1.5-pro")
    else:
        model_name = "gpt-4o-mini"

    model = llm.get_model(model_name)

    while attempts < max_attempts:
        print(f"\nThis is attempt number: {attempts}.")
        history_str = " | ".join(guess_history)

        prompt_text = f"""{instructions}. {objective}.
Based on previous attempts: {history_str}
Only return one 5-letter guess word.
"""
        guess_response = model.prompt(prompt_text)
        guess = extract_word(guess_response.text()).lower()

        if not check_word_validity(guess) or len(guess) != 5:
            print(f"Invalid guess '{guess}'. Trying next attempt.")
            attempts += 1
            continue

        attempts += 1
        GYs, feedback_details = colorize_guess(guess, target)
        print("Feedback on your guess:\n", feedback_details)

        guess_history.append(f"Attempt {attempts}: {guess} - {GYs}")
        results.append({
            "Global attempt #": run_id,
            "Run #": attempts,
            "LLM type": llm_type,
            "Target word": target,
            "Guessed word": guess,
            "Number of 'G' in colorised results": GYs.count('G'),
            "Number of 'Y' in colorised results": GYs.count('Y'),
            "Feedback": feedback_details
        })

        if guess == target:
            print(f"Correct! The word was '{target}'.")
            break

    if guess != target:
        print(f"Ran out of attempts; the word was '{target}'.")

def main():
    runs = int(input("Enter the number of runs: "))
    attempts_per_llm = 10
    results = []
    llm_types = ['chatgpt-4o-latest', 'claude-3-5-sonnet-latest', 'deepseek-reasoner', 'gemini-2.0-flash-thinking-exp-01-21', 'o1']  # add others as needed

    for run_id in range(1, runs + 1):
        for lt in llm_types:
            print(f"\n\nStarting run #{run_id} using {lt}")
            for _ in range(attempts_per_llm):
                play_wordle('puzzles/wordle.txt', run_id, lt, results)

    os.makedirs('results', exist_ok=True)
    with open('results/results_wordle.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("All runs completed. Results stored in 'results/results_wordle.json'.")

if __name__ == '__main__':
    main()
