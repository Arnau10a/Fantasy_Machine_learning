import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

# Registrem l'entorn
gym.register(
    id='Fantasy-v0',
    entry_point='FantasyEnv:FantasyEnv'
)

# Creem l'entorn
env = gym.make('Fantasy-v0')

# Verifiquem que l'entorn compleix amb l'API de Gymnasium
check_env(env)

# Definim paràmetres d'entrenament
TOTAL_TIMESTEPS = 200000
LEARNING_RATE = 0.0001
BATCH_SIZE = 128

# Entrenem amb DQN
def train_dqn():
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        buffer_size=10000,
        learning_starts=1000,
        verbose=1
    )
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save("fantasy_dqn")
    return model

# Funció d'avaluació
def evaluate_model(model, num_episodes=10):
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=num_episodes,
        deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward

if __name__ == "__main__":
    # Entrenem i avaluem DQN
    print("\nEntrenant DQN...")
    dqn_model = train_dqn()
    print("\nAvaluant DQN...")
    dqn_reward, dqn_std = evaluate_model(dqn_model)
    
    # Mostrem resultats
    print("\nResultats finals:")
    print(f"DQN: {dqn_reward:.2f} +/- {dqn_std:.2f}")
