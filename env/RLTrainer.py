import torch
import os
import torch.nn as nn
import pandas as pd
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from env.SurrogateModel import SurrogateModel
from env.Decoder import Decoder
from env.VQVAE_environment import VQVAE_Env, RenderCallback
from stable_baselines3.common.vec_env import DummyVecEnv


class Trainer:

    def __init__(
        self,
        surrogate_path,
        codebook_path,
        decoder_config,
        env_config,
        model_config,
        log_config,
    ):

        self.env_config = env_config
        self.model_config = model_config
        self.log_config = log_config
        self.decoder_config = decoder_config

        # Make sure the log directories exist
        self.current_path = os.getcwd()
        self.log_path = os.path.join(self.current_path, model_config["tensorboard_log"])
        os.makedirs(self.log_path, exist_ok=True)

        self.tensorboard_log_path = os.path.join(self.log_path, "tensorboard")
        os.makedirs(self.tensorboard_log_path, exist_ok=True)

        self.model_save_path = os.path.join(self.log_path, "models")
        os.makedirs(self.model_save_path, exist_ok=True)

        # Load Surrogate Model and Codebook
        self.surrogate_model = SurrogateModel(surrogate_path)
        self.codebook = torch.load(
            codebook_path, map_location="cpu"
        )  # Change this if you want to run on other device than CPU
        print("Codebook loaded from: ", codebook_path)

        # Load Decoder
        self.decoder_model = Decoder(
            out_dim=decoder_config["out_dim"],
            embed_dim=decoder_config["embed_dim"],
            h_nodes=decoder_config["h_nodes"],
            dropout=decoder_config["dropout"],
            scale=decoder_config["scale"],
            num_layers=decoder_config["num_layers"],
            load_path=decoder_config["load_path"],
        )

        # Load Environment
        self.check_env()

        # Create Environment
        self.env = DummyVecEnv([self.make_env])

    def train(self):

        # Setup Logging Configuration
        combined_config = {
            "model_config": self.model_config,
            "log_config": self.log_config,
            "env_config": self.env_config,
            "decoder_config": self.decoder_config,
        }
        # render_callback = RenderCallback(render_every=1)
        wandb_run = wandb.init(
            project=self.log_config["project"],
            # entity=self.log_config["entity"],
            config=combined_config,
            sync_tensorboard=self.log_config["sync_tensorboard"],
            save_code=self.log_config["save_code"],
        )

        # Setup Model
        if self.model_config["model"] == "PPO":
            self.model = PPO(
                policy=self.model_config["policy"],
                env=self.env,
                verbose=self.model_config["verbose"],
                tensorboard_log=self.tensorboard_log_path,
                n_steps=self.model_config["n_steps"],
                n_epochs=self.model_config["n_epochs"],
                batch_size=self.model_config["batch_size"],
            )
        elif self.model_config["model"] == "A2C":
            self.model = A2C(
                policy=self.model_config["policy"],
                env=self.env,
                verbose=self.model_config["verbose"],
                tensorboard_log=self.tensorboard_log_path,
                n_steps=self.model_config["n_steps"],
            )

        # Reset env
        self.env.reset()

        self.model.learn(
            total_timesteps=self.model_config["total_timesteps"],
            callback=[
                # render_callback,
                WandbCallback(
                    model_save_path=f"{self.model_save_path}/{wandb_run.id}",
                    gradient_save_freq=self.log_config["gradient_save_freq"],
                    verbose=self.log_config["verbose"],
                ),
            ],
            progress_bar=self.model_config["progress_bar"],
        )

        wandb_run.finish()

    def make_env(self):
        env = VQVAE_Env(
            embed_dim=self.env_config["embed_dim"],
            num_embeddings=self.env_config["num_embeddings"],
            max_allowed_actions=self.env_config["max_allowed_actions"],
            surrogate_model=self.surrogate_model,
            decoder=self.decoder_model,
            codebook=self.codebook,
            consider_previous_actions=self.env_config["consider_previous_actions"],
            num_previous_actions=self.env_config["num_previous_actions"],
            render_mode=self.env_config["render_mode"],
            render_data=self.env_config["render_data"],
            render_labels=self.env_config["render_labels"],
            log_dir=self.env_config["render_log_dir"],
        )

        env = Monitor(env)
        return env

    def check_env(self):
        env = self.make_env()
        check_env(env)
        print("Environment check passed")

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, load_path):
        if self.model_config["model"] == "PPO":
            self.model = PPO.load(load_path)
        elif self.model_config["model"] == "A2C":
            self.model = A2C.load(load_path)

    def evaluate_accuracy(self, num_episodes=10, num_steps=500, find_max=False):
        max_accuracy = float("-inf")
        state_at_max_accuracy = None

        accuracy = []
        for i in range(num_episodes):
            obs = self.env.reset()
            done = False
            max_reward = 0
            max_action = None
            max_obs_state = None
            cum_reward = 0
            print(f"Episode {i}")
            for j in range(num_steps):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, done, info = self.env.step(action)
                cum_reward += rewards
                print(
                    f"reward: {rewards}, action: {action}, cum_reward: {cum_reward}, max_reward: {max_reward}"
                )
                if max_reward < cum_reward:
                    max_reward = cum_reward
                    max_action = action
                    max_obs_state = obs["latent_vector"]

                if done:
                    print(
                        f"Episode {i},{j}: cum reward: {cum_reward}, max reward: {max_reward} action: {max_action} last action: {action}"
                    )
                    break
            if self.env_config["consider_previous_actions"]:
                rl_output = torch.tensor(obs["latent_vector"], dtype=torch.float32)
            else:
                rl_output = torch.tensor(obs, dtype=torch.float32)

            print(obs["latent_vector"] == max_obs_state)

            decoder_output = self.decoder_model(rl_output)
            calculate_accuracy = self.surrogate_model.evaluate(decoder_output)[0]
            accuracy.append(calculate_accuracy)

            if find_max and calculate_accuracy > max_accuracy:
                max_accuracy = calculate_accuracy
                state_at_max_accuracy = decoder_output

            print(f"Episode {i}: Accuracy: {calculate_accuracy}")

        print(f"Average Accuracy: {np.mean(accuracy)}")

        if find_max:
            print(f"Max Accuracy: {max_accuracy}")
            return state_at_max_accuracy