import os
import re
import json
import enchant
import llm
from dotenv import load_dotenv
load_dotenv()

with open('info.json', 'r') as file:
    data = json.load(file)

instructions = data.get('instructions_wg')
small_change = data.get('small_change_wg')
ATTEMPTS = 5
TURNS = 3

def get_llm_response(input_str, model_name):
    """
    Use 'llm' package to get a response from the chosen model.
    """
    model = llm.get_model(model_name)  # No more if/else mapping
    response = model.prompt(input_str).text()
    print(f"\nRaw LLM response from {model_name}:\n{response}\n")  # Debugging output
    return response

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
    """
    Preprocess raw LLM responses to remove unwanted formatting and ensure JSON compatibility.
    """
    if not response:
        return None

    # Remove Markdown-style JSON formatting
    response = re.sub(r"```json\s*([\s\S]*?)\s*```", r"\1", response, flags=re.DOTALL)

    # Remove any leading/trailing non-JSON text
    response = response.strip()

    # Remove trailing commas before brackets/braces
    response = re.sub(r',(?=\s*[\]])', '', response)

    # Replace single quotes with double quotes
    response = response.replace("'", '"')

    # Remove extra whitespace
    response = re.sub(r'\s+', ' ', response).strip()

    # Ensure it looks like JSON before parsing
    if not (response.startswith("{") or response.startswith("[")):
        print(f"Warning: Response does not start with JSON object/array:\n{response}\n")
        return None  

    return response

def extract_words_from_matrix(response):
    """
    Extract words from a JSON-like response, handling bad formatting.
    """
    print(f"Raw response before processing: {response}")

    response = preprocess_json_string(response)
    
    if not response:
        print("⚠️ Skipping empty or invalid response.")
        return []

    try:
        response_json = json.loads(response)
    except json.JSONDecodeError:
        print(f"❌ JSON parsing failed! Response was:\n{response}\n")
        return []

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
    elif isinstance(response_json, list):
        words = response_json
    else:
        # Maybe response was raw CSV
        words = [w.strip() for w in response.split(",")]

    words = [w.strip().replace('"','').replace("'", "") for w in words]
    return words

def check_word_validity(words):
    """
    Validate the words: check dictionary existence, length, and constraints.
    """
    if not words:
        print("⚠️ No words provided for validation.")
        return {}

    d = enchant.Dict("en_US")
    words_validity = {}

    for i, w in enumerate(words):
        valid = d.check(w.lower())  # Check if word exists in dictionary

        # Additional constraints
        if i == 0 and not w.startswith("C"):
            valid = False
        if i == len(words) - 1 and not w.endswith("N"):
            valid = False

        words_validity[w] = valid

    # Check for uniform word length
    lengths = [len(w) for w in words]
    if len(set(lengths)) > 1:
        print("❌ Word lengths are inconsistent.")
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
    objective_keys = ['objective_5']
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
