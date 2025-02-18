import os
import re
import json
import enchant
from dotenv import load_dotenv
load_dotenv()

import llm  # <-- new import for the "llm" package

with open('info.json', 'r') as file:
    data = json.load(file)

instructions = data.get('instructions_wg')
small_change = data.get('small_change_wg')
ATTEMPTS = 50
TURNS = 5

def get_llm_response(input_str, model_name):
    """
    Use 'llm' package to get a response from the chosen model.
    """
    model = llm.get_model(model_name)
    response = model.prompt(input_str)
    return response.text()

def create_word_matrix(objective, llm_type):
    """
    Generate a matrix of words with the new 'llm' approach.
    The words have to be valid English words (rows & columns),
    first word starts with 'C', last ends with 'N'.
    Return JSON-ish text to parse.
    """
    # Map our llm_type string to an actual model name
    if llm_type == 'openai':
        model_name = data.get("GPT_MODEL", "gpt-4o-mini")
    elif llm_type == 'claude':
        model_name = data.get("CLAUDE", "claude-3-5-sonnet-20240620")
    elif llm_type == 'groq':
        model_name = "groq-model"
    elif llm_type == 'gemini':
        model_name = data.get("GEMINI", "gemini-1.5-pro")
    else:
        model_name = data.get("GPT_MODEL", "gpt-4o-mini")

    prompt = f"""{instructions}. Objective is: {objective}.
The words have to be valid English words when read across rows and also down the columns.
Reply as JSON, for example:
'''
Word, Word, Word...
'''
"""
    return get_llm_response(prompt, model_name)

def regenerate_invalid_words(invalid_words, original_matrix, objective, llm_type):
    """
    Regenerate only the invalid words. 
    Use the partial context from the original matrix.
    """
    if llm_type == 'openai':
        model_name = data.get("GPT_MODEL", "gpt-4o-mini")
    elif llm_type == 'claude':
        model_name = data.get("CLAUDE", "claude-3-5-sonnet-20240620")
    elif llm_type == 'groq':
        model_name = "groq-model"
    elif llm_type == 'gemini':
        model_name = data.get("GEMINI", "gemini-1.5-pro")
    else:
        model_name = data.get("GPT_MODEL", "gpt-4o-mini")

    regeneration_prompt = f"""
{small_change}. You had generated an original matrix of words:
{original_matrix}
But this contained invalid words {invalid_words} when read across rows and columns.
Let's fix this. Objective is: {objective}.
The words must be valid English words (rows & columns). 
Reply with the final list in the same format:
'''
Word, Word, ...
'''
"""
    return get_llm_response(regeneration_prompt, model_name)

def preprocess_json_string(response):
    """
    Preprocess the response string to fix common JSON formatting issues.
    """
    if isinstance(response, str):
        # remove trailing commas before bracket/brace
        response = re.sub(r',(?=\s*[\]])', '', response)
        # replace single quotes with double quotes (basic approach)
        response = response.replace("'", '"')
        response = re.sub(r'\s+', ' ', response).strip()
    return response

def extract_words_from_matrix(response):
    """
    Extract words from a JSON-ish response
    (we're expecting something like {"words": ["CAT", "ARE", ...]} or plain text).
    """
    print(f"Raw response is: {response}")
    if isinstance(response, dict):
        response_json = response
    else:
        response = preprocess_json_string(response)
        try:
            response_json = json.loads(response)
        except json.JSONDecodeError:
            # fallback if we can't parse JSON
            response_json = {}

    words = []
    if isinstance(response_json, dict) and response_json:
        # Try to find a list in the dict
        for val in response_json.values():
            if isinstance(val, list):
                words = val
                break
            elif isinstance(val, str):
                words = val.split(",")
                break
    else:
        # maybe response was raw CSV
        if isinstance(response, str):
            words = [w.strip() for w in response.split(",")]

    words = [w.strip().replace('"','').replace("'", "") for w in words]
    return words

def check_word_validity(words):
    """
    Check that each word is in the dictionary,
    first word starts with 'C', last ends with 'N', all same length, etc.
    """
    d = enchant.Dict("en_US")
    words_validity = {}
    for i, w in enumerate(words):
        valid = d.check(w.lower())
        # Additional constraints
        if i == 0 and not w.startswith("C"):
            valid = False
        if i == len(words)-1 and not w.endswith("N"):
            valid = False
        words_validity[w] = valid

    # Check uniform length
    lengths = [len(w) for w in words]
    if len(set(lengths)) > 1:
        # Mark them all invalid if they're not uniform
        for w in words_validity:
            words_validity[w] = False

    invalid_words_count = sum(not x for x in words_validity.values())
    print(f"Validity measurement is {words_validity}")
    print(f"Number of invalid words: {invalid_words_count}\n\n")
    return words_validity

def main(attempt_number, objective, llm_type):
    """
    Single run to see if we can get a valid word grid within TURNS attempts.
    """
    results = {
        'attempt_number': attempt_number,
        'llm_type': llm_type,
        'runs': [],
        'success': False
    }
    original_matrix = None
    invalid_words_list = []

    for attempt_idx in range(1, TURNS + 1):
        attempt_data = {
            'index': attempt_idx,
            'matrix': None,
            'word_responses': None,
            'false_count': None,
            'error': None
        }
        try:
            if attempt_idx == 1:
                response = create_word_matrix(objective, llm_type)
            else:
                response = regenerate_invalid_words(invalid_words_list, original_matrix, objective, llm_type)

            if not response:
                raise ValueError("Empty response from LLM")

            words = extract_words_from_matrix(response)
            validity_dict = check_word_validity(words)
            invalid_count = sum(not v for v in validity_dict.values())

            attempt_data['matrix'] = words
            attempt_data['word_responses'] = list(validity_dict.keys())
            attempt_data['false_count'] = invalid_count

            invalid_words_list = [w for w, ok in validity_dict.items() if not ok]

            if invalid_count == 0:
                results['success'] = True
                # All valid, break
            else:
                original_matrix = response

        except ValueError as ve:
            attempt_data['error'] = f"ValueError: {ve}"
        except Exception as e:
            attempt_data['error'] = f"Exception: {e}"

        results['runs'].append(attempt_data)
        if results['success']:
            break

    if not results['success']:
        print("Failed to get a valid matrix within the maximum turn limit.")

    return results

def repeatedly_run_main():
    objective_keys = ['objective_3', 'objective_4', 'objective_5']
    llm_types = ['chatgpt-4o-latest', 'claude-3-5-sonnet-latest', 'deepseek-reasoner', 'gemini-2.0-flash-thinking-exp-01-21', 'o1']  # add others as needed
    all_results = {}

    for llm_type in llm_types:
        for ok in objective_keys:
            objective_text = data.get(ok)
            run_results = []
            for attempt_num in range(1, ATTEMPTS + 1):
                print(f"\nAttempt {attempt_num} for {ok} using {llm_type}...")
                res = main(attempt_num, objective_text, llm_type)
                run_results.append(res)
                if res['success']:
                    print("Success! Breaking early.")
                    break

            all_results[f"{ok}_{llm_type}"] = run_results
            # Save partial results
            with open(f"results_{ok}_{llm_type}.json","w") as f:
                json.dump(run_results, f, indent=4)

    # Combine final results
    os.makedirs('results', exist_ok=True)
    combined_results = {}
    for llm_type in llm_types:
        combined_results[llm_type] = {}
        for ok in objective_keys:
            file_path = f'results_{ok}_{llm_type}.json'
            with open(file_path, 'r') as file:
                data_j = json.load(file)
            combined_results[llm_type][f"matrix_{ok}"] = data_j

    with open('results/results_wg.json','w') as f:
        json.dump(combined_results, f, indent=4)
    print("Done collecting all results into 'results/results_wg.json'.")

if __name__ == "__main__":
    repeatedly_run_main()
