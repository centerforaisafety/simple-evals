#!/usr/bin/env python3
"""
Machiavelli Evaluation Script
"""

import asyncio
import json
import logging
import sys
import fire
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Check if game_data exists, download if not
game_data_dir = Path(__file__).parent / "game_data"
if not game_data_dir.exists():
    print("Downloading game_data (~3GB)...")
    import subprocess
    import zipfile
    script_dir = Path(__file__).parent
    zip_path = script_dir / "game_data.zip"
    
    subprocess.run([
        "wget", "-O", str(zip_path),
        "https://huggingface.co/datasets/justinphan3110/machiavelli/resolve/main/game_data.zip"
    ], check=True)
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(script_dir)
    zip_path.unlink()
    print("✓ game_data ready")

from machiavelli.agent.machiavelli_agents import MachiavelliAgent
from machiavelli.config import MachiavelliConfig
from machiavelli.game.machiavelli_env import MachiavelliEnv

sys.path.append(str(Path(__file__).parent.parent))
from shared import get_llm_agent_class, get_agent_config, sum_token_usage, get_max_token_usage
from evaluate_trajectories import load_results, print_evaluation_results

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logging.getLogger("main").setLevel(logging.INFO)


async def run_single_game(game: str,
                         model_name: str,
                         agent_config: dict,
                         output_dir: Path,
                         max_traj_length: int = 150,
                         debug: bool = False):
    """Run a single game and save trajectory."""
    env = MachiavelliEnv(game=game, model_name=model_name)
    
    llm_agent = get_llm_agent_class(**agent_config)
    agent = MachiavelliAgent(llm_agent)
    
    env = await agent.run_env(env, max_traj_length=max_traj_length, debug=debug)
    
    # Save trajectory
    output_dir.mkdir(parents=True, exist_ok=True)
    traj_file = output_dir / "trajectory.pkl"
    env.trajectory.save(traj_file)
    
    # Save logs
    logs_file = output_dir / "logs.json"
    token_usage = {
        'all_token_usage': llm_agent.all_token_usage.model_dump(),
        'max_token_usage': llm_agent.max_token_usage.model_dump()
    }
    with open(logs_file, 'w') as f:
        json.dump({
            'model_name': model_name,
            'agent_config': agent_config,
            'token_usage': token_usage,
            'history': agent.history
        }, f, indent=4)
    
    points = env.trajectory.get_points(normalize=True)
    achievements = env.trajectory.get_achievements()
    steps = len(env.trajectory)
    
    print(f"Game: {env.game} | Points: {points} | Steps: {steps} | Achievements: {achievements}")
    
    env.close()
    
    return llm_agent.all_token_usage


async def generate_trajectories(games: list,
                                model_name: str,
                                agent_config: dict,
                                output_dir: Path,
                                max_concurrent: int = 5,
                                max_traj_length: int = 150,
                                debug: bool = False):
    """Generate game trajectories for all games."""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_game_with_semaphore(game):
        async with semaphore:
            return await run_single_game(
                game=game,
                model_name=model_name,
                agent_config=agent_config,
                output_dir=output_dir / game,
                max_traj_length=max_traj_length,
                debug=debug
            )
    
    tasks = [run_game_with_semaphore(game) for game in games]
    
    all_token_usages = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Running games"):
        token_usage = await task
        all_token_usages.append(token_usage)
    
    return all_token_usages


def run_eval(model: str,
             output_dir: str,
             models_config: str = "configs/models.yaml",
             max_concurrent: int = 5,
             games: str = None,
             debug: bool = False,
             skip_play: bool = False,
             max_traj_length: int = 150):
    """
    Run Machiavelli evaluation.
    
    Args:
        model: Model name from models.yaml
        output_dir: Directory to save trajectories and results (required)
        models_config: Path to models configuration file
        max_concurrent: Maximum number of concurrent games
        games: Comma-separated list of games (e.g., "avatar-of-the-wolf,kung-fu") or None for all
        debug: Enable debug mode
        skip_play: Skip playing games and only evaluate existing trajectories
        max_traj_length: Maximum trajectory length per game
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model configuration
    agent_config = get_agent_config(model, models_config)
    print(f"Using model: {model}")
    
    # Determine games to run
    if games:
        games_list = games.split(',')
    else:
        games_list = MachiavelliConfig().games_test
    
    print(f"Running {len(games_list)} games")
    print(f"Output directory: {output_path}")
    
    async def run_evaluation():
        all_token_usages = []
        
        if not skip_play:
            # Step 1: Generate trajectories
            print("\n=== Step 1: Generating Trajectories ===")
            all_token_usages = await generate_trajectories(
                games=games_list,
                model_name=model,
                agent_config=agent_config,
                output_dir=output_path,
                max_concurrent=max_concurrent,
                max_traj_length=max_traj_length,
                debug=debug
            )
        else:
            print("Skipping trajectory generation (skip_play=True)")
        
        # Step 2: Evaluate trajectories
        print("\n=== Step 2: Evaluating Trajectories ===")
        df = load_results(str(output_path))
        
        results_file = str(output_path / f"{model.replace('/', '_')}_results.csv")
        print_evaluation_results(df, results_file)
        
        # Print token usage summary
        if all_token_usages:
            total_token_usage = sum_token_usage(all_token_usages)
            max_token_usage = get_max_token_usage(all_token_usages)
            
            print("\n===== Token Usage =====")
            print(f"Total: {total_token_usage.model_dump()}")
            print(f"Max per game: {max_token_usage.model_dump()}")
        
        print(f"\nResults saved to {results_file}")
    
    asyncio.run(run_evaluation())


if __name__ == '__main__':
    fire.Fire(run_eval)
