import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    # Load JSON data
    file_path = 'results/results_wordle.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Ensure the columns we need exist
    required_cols = [
        'Global attempt #', 'Run #', 'LLM type', 
        'Target word', 'Guessed word',
        "Number of 'G' in colorised results", "Number of 'Y' in colorised results"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"Missing columns in results_wordle.json: {missing}")
        return

    # 1) Mark whether each guess is a success
    df['is_success'] = df['Guessed word'] == df['Target word']

    # 2) Count how many total lines (guesses) are successful
    total_successes = df['is_success'].sum()
    total_guesses   = len(df)
    print(f"Total guesses: {total_guesses}, total successful guesses: {total_successes}")

    # 3) For each "Global attempt #" + "LLM type", we can figure out if that “game” ended in success or not.
    #    Because each game is repeated up to 5 attempts. We'll define a "game_id" = (LLM type, Global attempt #).
    df['game_id'] = df['LLM type'].astype(str) + "_" + df['Global attempt #'].astype(str)

    # We want to see how many attempts it took to succeed (or fail):
    # For each game, if success occurred, it happened on the minimal "Run #" where is_success==True.
    # If no success, we set attempts_used = max run # in that game (which is presumably 5).

    # 4) Compute per-game stats
    #    - attempts_used
    #    - success or not
    # We'll group by "game_id".
    game_stats = []
    for gid, group in df.groupby('game_id'):
        model_name = group['LLM type'].iloc[0]
        global_attempt = group['Global attempt #'].iloc[0]

        # Did we ever guess right in this game?
        success_rows = group[group['is_success']]
        if len(success_rows) > 0:
            # the earliest success attempt #:
            success_run_number = success_rows['Run #'].min()
            attempts_used = success_run_number
            success_flag = True
        else:
            # no success: attempts used is the max run # in that game
            attempts_used = group['Run #'].max()
            success_flag = False

        game_stats.append({
            'game_id': gid,
            'LLM type': model_name,
            'Global attempt': global_attempt,
            'attempts_used': attempts_used,
            'success': success_flag,
        })

    game_stats_df = pd.DataFrame(game_stats)

    # 5) Evaluate success rates by model
    model_group = game_stats_df.groupby('LLM type')
    success_rate_by_model = model_group['success'].mean().sort_values(ascending=False)
    print("\nSuccess Rate by Model:\n", success_rate_by_model)

    # 6) Evaluate average attempts by model (on successful games only)
    successful_games_df = game_stats_df[game_stats_df['success']]
    avg_attempts_by_model = successful_games_df.groupby('LLM type')['attempts_used'].mean().sort_values()
    print("\nAverage Attempts (Successful Only) by Model:\n", avg_attempts_by_model)

    # ========== Plot 1: Success Rate by Model (Bar Chart) ==========
    plt.figure(figsize=(8, 5))
    sns.barplot(x=success_rate_by_model.index, y=success_rate_by_model.values, palette='Blues_d')
    plt.ylim(0,1.05)
    plt.title("Wordle Success Rate by Model")
    plt.ylabel("Success Rate")
    plt.xlabel("Model")
    for i, v in enumerate(success_rate_by_model.values):
        plt.text(i, v+0.01, f"{v:.1%}", ha='center', fontweight='bold')
    plt.tight_layout()
    os.makedirs('charts', exist_ok=True)
    plt.savefig('charts/wordle_success_rate_by_model.png')
    # plt.show()

    # ========== Plot 2: Avg Attempts by Model (Successful Games) ==========
    plt.figure(figsize=(8, 5))
    sns.barplot(x=avg_attempts_by_model.index, y=avg_attempts_by_model.values, palette='Greens_d')
    plt.title("Average Attempts (Successful Games) by Model")
    plt.ylabel("Avg # of Attempts")
    plt.xlabel("Model")
    for i, v in enumerate(avg_attempts_by_model.values):
        plt.text(i, v+0.05, f"{v:.2f}", ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('charts/wordle_avg_attempts_success_by_model.png')
    # plt.show()

    # ========== Retain old G/Y trend logic if desired ==========
    # For reference, the original code was grouping by "Run #", but that might not isolate each game.
    # We'll keep it for historical reasons, but if you want to group by model + run #, you can do so.

    # Convert numeric columns just in case
    df["Number of 'G' in colorised results"] = pd.to_numeric(df["Number of 'G' in colorised results"], errors='coerce')
    df["Number of 'Y' in colorised results"] = pd.to_numeric(df["Number of 'Y' in colorised results"], errors='coerce')

    # Let's do a quick lineplot showing G and Y across attempts, split by model
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x="Run #", 
        y="Number of 'G' in colorised results", 
        hue="LLM type", 
        ci=None, 
        marker='o'
    )
    plt.title("G counts per Attempt #, by Model")
    plt.ylim(0, 5)
    plt.savefig("charts/wordle_G_trends_by_model.png")
    # plt.show()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df, 
        x="Run #", 
        y="Number of 'Y' in colorised results", 
        hue="LLM type", 
        ci=None, 
        marker='o'
    )
    plt.title("Y counts per Attempt #, by Model")
    plt.ylim(0, 5)
    plt.savefig("charts/wordle_Y_trends_by_model.png")
    # plt.show()

    print("\nAnalysis complete. Charts have been saved in the 'charts/' folder.")

if __name__ == "__main__":
    main()
