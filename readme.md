# AI Snake Game using Stable-Baselines3 (PPO)

This project implements an AI agent that learns to play the classic Snake game using the Proximal Policy Optimization (PPO) algorithm from the `stable-baselines3` library and a custom `gymnasium` environment.

## Goal

To train a robust AI agent capable of achieving high scores in the Snake game by navigating effectively to eat food while avoiding collisions. This implementation uses industry-standard RL libraries relevant for professional AI/ML roles.

## How it Works

1.  **Environment (`snake_env.py`):**
    *   A custom environment built using `Pygame` for the core logic and rendering.
    *   Adheres to the `gymnasium.Env` API standard, making it compatible with libraries like `stable-baselines3`.
    *   **Observation Space:** A 11-dimensional vector representing the snake's immediate surroundings, direction, and relative food location (see `snake_env.py` for details).
    *   **Action Space:** Discrete(3) - representing relative actions: 0 (Straight), 1 (Right Turn), 2 (Left Turn).
    *   **Reward Shaping:** Includes rewards for eating food (+10), penalties for dying (-10), and small rewards/penalties for moving closer/further from the food to encourage exploration and efficient pathfinding.

2.  **Agent (PPO via `stable-baselines3`):**
    *   Uses the **Proximal Policy Optimization (PPO)** algorithm, a state-of-the-art actor-critic method known for its stability and performance across various tasks.
    *   The agent's policy (how it chooses actions) and value function (how it estimates state value) are represented by Multi-Layer Perceptrons (MLPs) defined by `MlpPolicy`.
    *   `stable-baselines3` handles the complex training loop, data collection, policy updates, and optimizations.

3.  **Training (`train_sb3.py`):**
    *   Instantiates the custom `SnakeEnv`.
    *   Wraps the environment with `Monitor` for logging episode rewards and lengths.
    *   Uses `EvalCallback` to periodically evaluate the agent's performance on separate episodes and save the best-performing model based on average reward.
    *   Logs training progress to TensorBoard for visualization.
    *   Trains the PPO model for a specified number of `TOTAL_TIMESTEPS`.
    *   Saves the final trained model.

4.  **Testing (`test_sb3.py`):**
    *   Loads a pre-trained model (either the best one or the final one).
    *   Runs the agent in the environment with rendering enabled (`render_mode="human"`).
    *   Uses deterministic actions (`deterministic=True`) for evaluation to see the agent's learned policy without exploration noise.

## Skills Demonstrated

*   **Python:** Core programming language.
*   **Reinforcement Learning:** Environment design (Gymnasium API), Proximal Policy Optimization (PPO), Reward Shaping, understanding of RL training loops and evaluation.
*   **AI/ML Libraries:** Stable-Baselines3, PyTorch (as backend), Gymnasium, NumPy.
*   **Deep Learning:** Understanding of policy/value networks (MLPs).
*   **Software Engineering:** Modular code structure (Env vs Train vs Test), use of standard libraries, logging (TensorBoard).
*   **Game Development:** Basic game logic and rendering using `Pygame`.

## Requirements

*   Python 3.8+
*   `pip install gymnasium pygame stable-baselines3[extra] torch numpy`

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/AISnakeGame_SB3.git # Recommend renaming repo
    cd AISnakeGame_SB3
    ```
    *(Replace YourUsername and potentially AISnakeGame_SB3)*
2.  **Set up environment and install dependencies:**
    *(Using venv)*
    ```bash
    python -m venv venv
    # Activate environment (Windows PowerShell)
    .\venv\Scripts\Activate.ps1
    # OR (MacOS/Linux)
    # source venv/bin/activate
    pip install gymnasium pygame stable-baselines3[extra] torch numpy
    ```
    *(Or use Conda if preferred)*
3.  **Train the agent:**
    ```bash
    python train_sb3.py
    ```
    *   Training progress will be printed to the console.
    *   Logs will be saved in the `logs/` directory. View them with `tensorboard --logdir=./logs/`.
    *   The best model will be saved in `models/best_model.zip`, and the final model as `models/ppo_snake.zip`. Training can take time (millions of timesteps for good performance).
4.  **Test the trained agent:**
    ```bash
    python test_sb3.py
    ```
    *   This will load the saved model (defaults to `best_model.zip`) and run several episodes with the game window visible.

## Next Steps / Potential Improvements

*   Implement a CNN-based policy (`CnnPolicy` in SB3) using screen pixels or a grid representation as input instead of the feature vector. This requires modifying the `observation_space` and `_get_obs` method in `snake_env.py`.
*   Extensive hyperparameter tuning for PPO (learning rate, batch size, gamma, n_steps, etc.).
*   Experiment with other algorithms available in Stable Baselines3 (e.g., DQN, A2C).
*   Implement curriculum learning (start with a smaller grid, gradually increase size).