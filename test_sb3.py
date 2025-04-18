# test_sb3.py
import time
import pygame # Import pygame to handle events directly
from stable_baselines3 import PPO
from snake_env import SnakeEnv # Import custom environment

# --- Configuration ---
MODEL_PATH = "models/best_model.zip" # Load the best model saved by EvalCallback
# MODEL_PATH = "models/ppo_snake.zip" # Or load the final model
N_TEST_EPISODES = 10

if __name__ == "__main__":
    print(f"Loading model from {MODEL_PATH}...")
    # Load the trained agent
    model = PPO.load(MODEL_PATH)

    print("Setting up test environment...")
    # Create the environment with human rendering enabled
    env = SnakeEnv(render_mode="human", grid_size=10) # Use same grid size as training

    print(f"Running {N_TEST_EPISODES} test episodes...")
    total_reward_sum = 0
    run = True # Flag to control the outer loop based on pygame events

    for episode in range(N_TEST_EPISODES):
        if not run: # Check if user quit in previous episode
            break

        obs, info = env.reset()
        terminated = False
        truncated = False # Gymnasium uses truncated
        episode_reward = 0
        step = 0
        while not terminated and not truncated and run: # Check run flag here too
            # --- Pygame Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("Quit event detected! Stopping testing.")
                    run = False # Set flag to exit outer loop
                    break # Exit event loop
            if not run:
                break # Exit while loop if quit event detected
            # -----------------------------

            # Use deterministic=True for evaluation (agent uses best known action)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            env.render() # Render the game window
            # Add a small delay to make it watchable, adjust as needed
            time.sleep(0.05)

        if run: # Only print results if the episode wasn't interrupted by quitting
            print(f"Episode {episode + 1}: Score = {info.get('score', 'N/A')}, Reward = {episode_reward:.2f}, Steps = {step}")
            total_reward_sum += episode_reward
            time.sleep(1) # Pause briefly between episodes

    print("-" * 20)
    # Calculate average only on completed episodes if interrupted
    completed_episodes = episode + 1 if run else episode
    if completed_episodes > 0:
         avg_reward = total_reward_sum / completed_episodes
         print(f"Average reward over {completed_episodes} completed episodes: {avg_reward:.2f}")
    else:
         print("No episodes completed.")
    print("-" * 20)

    # Close the environment
    env.close()
    print("Testing complete.")