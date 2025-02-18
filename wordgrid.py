import os
import re
import json
import llm
from dotenv import load_dotenv
load_dotenv()

with open('info.json', 'r') as file:
    data = json.load(file)

instructions = data.get('instructions_wg')
small_change = data.get('small_change_wg')
ATTEMPTS = 10
TURNS = 5

def get_llm_response(input_str, model_name):
    """
    Use 'llm' package to get a response from the chosen model.
    """
    model = llm.get_model(model_name)  # No more if/else mapping
    response = model.prompt(input_str)
    return response.text()

def create_word_matrix(objective, llm_type):
    """
    Generate a matrix of words with the 'llm' approach.
    The words have to be valid English words (rows & columns).
    """
    # Now we just treat llm_type as the actual model name:
    prompt = f"""{instructions}. Objective is: {objective}.
The words have to be valid English words when read across rows and also down the columns. As you place each word, verify that it forms valid words both horizontally and vertically. Double-check that all words in the grid are valid English words. Verify that the grid meets all specified constraints before finalizing your answer.

Your final output should be in JSON format, with each word representing a row in the grid. For example:

```json
["WORD", "WORD", "WORD", "WORD"]
```
"""
    return get_llm_response(prompt, llm_type)

def regenerate_invalid_words(invalid_words, original_matrix, objective, llm_type):
    """
    Regenerate only the invalid words, using the partial context.
    """
    regeneration_prompt = f"""
{small_change}. You had generated an original matrix of words:
{original_matrix}
But this contained invalid words {invalid_words} when read across rows and columns.
Let's fix this. Objective is: {objective}.
The words have to be valid English words when read across rows and also down the columns. As you place each word, verify that it forms valid words both horizontally and vertically. Double-check that all words in the grid are valid English words. Verify that the grid meets all specified constraints before finalizing your answer.

Your final output should be in JSON format, with each word representing a row in the grid. For example:

```json
["WORD", "WORD", "WORD", "WORD"]
```
"""
    return get_llm_response(regeneration_prompt, llm_type)

def preprocess_json_string(response):
    ...
    # (same as before)

def extract_words_from_matrix(response):
    ...
    # (same as before)

def check_word_validity(words):
    ...
    # (same as before)

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
    objective_keys = ['objective_4', 'objective_5']
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
