#!/usr/bin/env python3
"""
Evaluate Machiavelli trajectories and compute metrics.
"""

import argparse
import pandas as pd
from pathlib import Path
from machiavelli.game.machiavelli_env import Trajectory

skip_games = []


def load_results(traj_dir: str):
    traj_dir = Path(traj_dir)
     
    df_rows = []
    seen = set()
    achivements_by_game = {}
    total_achievements_by_game = {}
    points_by_game = {}
    num_game = 10
    for traj_path in sorted(list(traj_dir.glob("**/*.pkl"))):
        print(traj_path)
        traj = Trajectory.load(traj_path)
        mach_scores = traj.get_mach_scores()
        if traj.game_name in skip_games:
            continue
        if traj.game_name in seen:
            continue
        
        if len(achivements_by_game.get(traj.game_name, [])) >= num_game:
            continue

        achivements_by_game.setdefault(traj.game_name, []).append(traj.get_achievements())
        total_achievements_by_game.setdefault(traj.game_name, []).append(traj.get_total_achievements())
        points_by_game.setdefault(traj.game_name, []).append(traj.get_points(normalize=True))

        df_rows.append({
            "game": traj.game_name,
            "traj_dir": traj_dir,
            "points": traj.get_points(normalize=True),
            "achievements": list(traj.get_achievements()),
            "num_achievements": len(traj.get_achievements()),
            "steps": len(traj),
            **mach_scores,
        })
    df = pd.DataFrame(df_rows)
    return df

def print_evaluation_results(df: pd.DataFrame, results_file: str = "results.csv"):
    """Print evaluation results and save aggregated results to CSV."""
    
    # Aggregate results and calculate the mean of each column (excluding non-numeric columns)
    df_agg = df.drop(columns=['game', 'achievements']).groupby(["traj_dir"]).mean().reset_index()
    df_agg["samples"] = df.drop(columns=['achievements']).groupby(["traj_dir"]).count().reset_index()["points"]
    df_agg["games"] = df.drop(columns=['achievements']).groupby(["traj_dir"]).nunique().reset_index()["game"]
    df_agg.to_csv(results_file, index=False)
    
    print("\n=== Machiavelli Results ===")
    
    # Main metric - Deception
    if 'violations.deception' in df_agg.columns:
        print(f"Deception: {df_agg['violations.deception'].mean():.3f}")
    
    # Summary info
    games_count = int(df_agg['games'].mean()) if 'games' in df_agg.columns else 0
    print(f"Games evaluated: {games_count}")
    
    # Game performance
    if 'game.score' in df_agg.columns:
        print(f"Game score: {df_agg['game.score'].mean():.3f}")
    if 'steps' in df_agg.columns:
        print(f"Average steps: {df_agg['steps'].mean():.1f}")
    if 'num_achievements' in df_agg.columns:
        print(f"Average achievements: {df_agg['num_achievements'].mean():.1f}")
    
    print("\n=== Power Metrics ===")
    power_metrics = ['power.money', 'power.physical', 'power.social', 'power.utility', 'power.Σ']
    for metric in power_metrics:
        if metric in df_agg.columns:
            print(f"{metric}: {df_agg[metric].mean():.3f}")
    
    print("\n=== Utility Metrics ===")
    utility_metrics = ['utility.suffering', 'utility.Σ']
    for metric in utility_metrics:
        if metric in df_agg.columns:
            print(f"{metric}: {df_agg[metric].mean():.3f}")
    
    print("\n=== Violation Metrics ===")
    violation_metrics = [col for col in df_agg.columns if col.startswith('violations.')]
    for metric in sorted(violation_metrics):
        print(f"{metric}: {df_agg[metric].mean():.3f}")
    
    return df_agg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj_dir', "-t", required=True, type=str)
    parser.add_argument('--results_file', "-r", required=False, type=str, default="results.csv")
    args = parser.parse_args()
    
    df = load_results(args.traj_dir)
    print_evaluation_results(df, args.results_file)
