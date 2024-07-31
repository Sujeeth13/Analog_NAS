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
from env.surrogate.SurrogateModel import SurrogateModel
from env.vqvae.decoder import Decoder
from env.environment import VQVAE_Env, RenderCallback
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
            codebook_path,
            map_location="cpu",  # TODO: Change this to a parameter in the future to allow for GPU training
        )
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

    def train(self, custom_callback=None):

        # Setup Logging Configuration
        combined_config = {
            "model_config": self.model_config,
            "log_config": self.log_config,
            "env_config": self.env_config,
            "decoder_config": self.decoder_config,
        }


        print('Initializing WanderDB')
        # render_callback = RenderCallback(render_every=1)
        wandb_run = wandb.init(
            project=self.log_config["project"],
            # entity=self.log_config["entity"],
            config=combined_config,
            sync_tensorboard=self.log_config["sync_tensorboard"],
            save_code=self.log_config["save_code"],
        )


        print('Setting up Model')
        print('Model Config:', self.model_config)
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
        print('Resetting Environment')
        self.env.reset()
        print('Training Model')
        if custom_callback:
            self.model.learn(
                total_timesteps=self.model_config["total_timesteps"],
                callback=[
                    custom_callback,
                    WandbCallback(
                        model_save_path=f"{self.model_save_path}/{wandb_run.id}",
                        gradient_save_freq=self.log_config["gradient_save_freq"],
                        verbose=self.log_config["verbose"],
                    ),
                ],
                progress_bar=self.model_config["progress_bar"],
            )
        else:
            print("Training Started")
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
            consider_max_params=self.env_config["consider_max_params"],
            max_params=self.env_config["max_params"],
            min_params=self.env_config["min_params"],
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

    def evaluate_accuracy(self, num_episodes=10, make_plot=False):

        max_accuracy = float("-inf")
        state_at_max_accuracy = None

        accuracy_list = []
        if make_plot:
            obs_list = []
            episode_len_list = []
            episode_len = 0

        for i in range(num_episodes):
            obs = self.env.reset()
            done = False
            max_reward = 0
            max_action = None
            max_obs_state = None
            cum_reward = 0
            if make_plot and episode_len != 0:
                episode_len_list.append(episode_len)
            episode_len = 0
            num_steps = 0
            print(f"Episode {i}")
            while not done:
                if make_plot:
                    if self.env_config["consider_previous_actions"] or self.env_config['consider_max_params']:
                        obs_list.append(obs["latent_vector"])
                    else:
                        obs_list.append(obs)
                    episode_len += 1
                action, _ = self.model.predict(obs, deterministic=True)
                # print("action:", action)

                obs, reward, done, info = self.env.step(
                    action
                )  # Note: Should this be obs, rewards, done, truncated, info = self.env.step(action) instead? --> nope as done = truncated or terminated  based on SB3 VecEnv documentation: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
                num_steps += 1

                # print("Reward Type:", type(reward))
                # print("rewards:", reward)
                # print("done:", done)

                cum_reward += reward
                cum_reward = (
                    cum_reward.item()
                    if isinstance(cum_reward, (np.ndarray, list))
                    else cum_reward
                )
                max_reward = (
                    max_reward.item()
                    if isinstance(max_reward, (np.ndarray, list))
                    else max_reward
                )
                # print("cum_reward:", cum_reward)

                # print(
                #     f"reward: {rewards}, action: {action}, cum_reward: {cum_reward}, max_reward: {max_reward}"
                # )

                if max_reward < cum_reward:
                    max_reward = cum_reward
                    max_action = action
                    if self.env_config["consider_previous_actions"]:
                        max_obs_state = obs[
                            "latent_vector"
                        ]  # TODO: Why is this not being used? Should it be used? or else we should just get rid of this part of the code
                    else:
                        max_obs_state = obs
                    # print(f"$$$Max Reward: {max_reward}, Max Action: {max_action}")

                if done:
                    print(
                        f"Episode {i},{num_steps}: cum reward: {cum_reward}, max reward: {max_reward}, max action: {max_action}, last action: {action}"
                    )
                    break
            # if self.env_config["consider_previous_actions"]:
            #     rl_output = torch.tensor(obs["latent_vector"], dtype=torch.float32)
            # else:
            #     rl_output = torch.tensor(obs, dtype=torch.float32)
            rl_output = torch.tensor(max_obs_state, dtype=torch.float32)

            decoder_output = self.decoder_model(rl_output)
            calculated_accuracy = self.surrogate_model.evaluate(decoder_output)[0]
            accuracy_list.append(calculated_accuracy)

            if calculated_accuracy > max_accuracy:
                max_accuracy = calculated_accuracy
                state_at_max_accuracy = decoder_output

            print(
                f"Episode {i}: Episode Accuracy: {calculated_accuracy}, Max Accuracy till Episode: {max_accuracy}"
            )

        print(f"Average Accuracy: {np.mean(accuracy_list)}")

        if make_plot:
            np.save("env/render/states.npy", np.array(obs_list))
            np.save("env/render/episode_lengths.npy", np.array(episode_len_list))
            self.env.render()

        return state_at_max_accuracy

    def calculate_accuracy_for_decoded_state(self, decoded_state):
        return self.surrogate_model.evaluate(decoded_state)[0]
