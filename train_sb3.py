# train_sb3.py
import os
from stable_baselines3 import PPO # Using PPO algorithm
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from snake_env import SnakeEnv # Import custom environment

# --- Configuration ---
LOG_DIR = "logs/"
MODEL_SAVE_PATH = "models/ppo_snake"
TOTAL_TIMESTEPS = 2_000_000 # Increase for better performance (e.g., 1M, 5M, 10M)
EVAL_FREQ = 10_000 # Evaluate model every N steps
EVAL_EPISODES = 10 # Number of episodes to average evaluation score over
STOP_REWARD_THRESHOLD = 50 # Stop training if avg reward over EVAL_EPISODES reaches this

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

if __name__ == "__main__":
    print("Setting up environment and training...")

    # Create the environment, wrap with Monitor for logging
    # Use make_vec_env for potential parallel environments later (n_envs=1 for now)
    env = make_vec_env(lambda: Monitor(SnakeEnv(grid_size=10)), n_envs=1) # Smaller grid might learn faster

    # Callbacks
    # Stop training when the model reaches the reward threshold
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=STOP_REWARD_THRESHOLD, verbose=1)
    # Evaluate the model periodically and save the best model
    eval_callback = EvalCallback(env, # Use the same vec env for evaluation
                                 callback_on_new_best=stop_callback, # Chain callbacks
                                 eval_freq=EVAL_FREQ,
                                 n_eval_episodes=EVAL_EPISODES,
                                 best_model_save_path=os.path.dirname(MODEL_SAVE_PATH), # Save in models/
                                 log_path=LOG_DIR, # Save eval logs
                                 deterministic=True, # Use deterministic actions for evaluation
                                 render=False, # Don't render during evaluation
                                 verbose=1)

    # Create the PPO agent
    # MlpPolicy uses a Multi-Layer Perceptron (good for vector observations)
    # See SB3 documentation for hyperparameter tuning
    model = PPO("MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=LOG_DIR,
                # Common hyperparameters to tune:
                # learning_rate=0.0003,
                # n_steps=2048, # Steps per rollout per environment
                # batch_size=64,
                # n_epochs=10,
                # gamma=0.99, # Discount factor
                # gae_lambda=0.95, # Factor for Generalized Advantage Estimation
                # ent_coef=0.0, # Entropy coefficient (regularization)
                # vf_coef=0.5 # Value function coefficient
               )

    # Check if a pre-trained model exists
    if os.path.exists(f"{MODEL_SAVE_PATH}.zip"):
        print(f"Loading existing model from {MODEL_SAVE_PATH}.zip")
        model = PPO.load(MODEL_SAVE_PATH, env=env, tensorboard_log=LOG_DIR) # Ensure env is reset
        # If you want to reset learning rate or other parameters when continuing:
        # model.learning_rate = 0.0001 # Example: Set a new learning rate
        print("Model loaded. Continuing training...")
    else:
        print("No existing model found. Starting training from scratch.")


    # Train the agent
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=eval_callback, # Use the evaluation callback
                    log_interval=1, # Log stats every episode/rollout
                    progress_bar=True) # Show a progress bar
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # Save the final model
    print(f"Saving final model to {MODEL_SAVE_PATH}.zip")
    model.save(MODEL_SAVE_PATH)

    print("Training complete.")
    print(f"Logs saved in: {LOG_DIR}")
    print(f"Best model saved in: {os.path.dirname(MODEL_SAVE_PATH)}/best_model.zip")
    print(f"Final model saved as: {MODEL_SAVE_PATH}.zip")

    # Close the environment
    env.close()

    print("\nTo view training logs, run: tensorboard --logdir=./logs/")