import os
import re
import json
import enchant
from dotenv import load_dotenv
load_dotenv()
import llm

from puzzles.sudokugen import generate_sudoku, is_valid_move, find_empty_location
from sudokusolve import encode_sudoku, decode_solution, transpose, solve_sudoku_with_explanation
from pysat.formula import CNF
from pysat.solvers import Glucose3

with open('info.json', 'r') as file:
    data = json.load(file)

instructions = data.get('instructions_s')
objective = data.get('objective_s')
ATTEMPTS = 10
THRESHOLD = 10

def create_sudoku(sudoku_board, objective_text):
    """Generate a sudoku solution guess using the 'llm' package."""
    model = llm.get_model("gpt-4o-mini")  # or use data.get("GPT_MODEL", "gpt-4o-mini") if desired
    prompt_text = f"""{instructions}. Objective is: {objective_text}.
Given the following sudoku matrix: {sudoku_board}
Reply with the numbers that fill in the puzzle, format like:
Number, Number, etc
"""
    response = model.prompt(prompt_text)
    return response.text()

def create_sudoku_row(sudoku_board, objective_text):
    """Generate a sudoku solution row-by-row using the 'llm' package."""
    model = llm.get_model("gpt-4o-mini")
    responses = []
    for row in sudoku_board:
        prompt_text = f"""{instructions}. Objective is: {objective_text}.
Given the sudoku puzzle: {sudoku_board}
Solve this row: {row}.
Answer in the format:
Number, Number, etc
"""
        resp = model.prompt(prompt_text)
        responses.append(resp.text())
    return responses

def parse_response_to_int_list(response):
    """Extract integers (digits 1-9) from the LLM's response."""
    numbers = re.findall(r'\d+', response.strip())
    return [int(num) for num in numbers]

def solve_sudoku(board, replacements):
    """Fill zeroes in `board` with a provided list of integers (`replacements`)."""
    replacement_queue = list(replacements)
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:  # Found a blank
                if replacement_queue:
                    board[i][j] = replacement_queue.pop(0)
                else:
                    return "Not enough replacement numbers provided."
    if replacement_queue:
        return "Too many replacement numbers provided."
    return board

def check_solution(sudoku):
    """Use the SAT solver to check or finalize the sudoku solution."""
    cnf = encode_sudoku(sudoku)
    solver = Glucose3()
    solver.append_formula(cnf)
    if solver.solve():
        model = solver.get_model()
        solution = decode_solution(model)
        solution = transpose(solution)  # Because we transposed row/col in decode_solution
        print("\nSAT Solver Solution")
        for row in solution:
            print(row)
    else:
        print("No solution found, sorry!")

def main():
    for puzzle_number in range(1, ATTEMPTS + 1):
        sudoku = generate_sudoku(puzzle_number)
        if puzzle_number <= THRESHOLD:
            # For puzzles <= THRESHOLD, you could do a direct approach or none at all:
            pass
        else:
            # For puzzle numbers greater than 10, solve row by row:
            print(f"\n--- Puzzle {puzzle_number} (Row by Row) ---\n")
            response_list = create_sudoku_row(sudoku, objective)
            # The LLM might give us row solutions in multiple lines;
            # combine them into one replacement list:
            combined_response = " ".join(response_list)
            replacements = parse_response_to_int_list(combined_response)
            solved_board = solve_sudoku(sudoku, replacements)
            print("\nProposed solution from the model:\n")
            for row in solved_board:
                print(row)
            check_solution(sudoku)

if __name__ == "__main__":
    main()
