import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque
import torch
from stable_baselines3.common.callbacks import BaseCallback
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import datetime
import os


class RenderCallback(BaseCallback):
    def __init__(self, render_every: int, verbose=0):
        super(RenderCallback, self).__init__(verbose)
        self.render_every = render_every
        self.episode_count = 0
        self.states = []  # Intialize the list to track states for an episode

    def _on_training_start(self) -> None:
        """Called before the first rollout starts."""
        # print("Training is starting...")
        self.states = []  # Ensure states list is empty at the start

    def _on_step(self) -> bool:
        # Track the next_state
        # print(self.locals)
        # print("step")
        next_state = self.locals[
            "obs_tensor"
        ]  # Assuming 'new_obs' contains the next state information
        next_state = next_state["latent_vector"]
        self.states.append(next_state.numpy())
        return True

    def _on_rollout_end(self):
        # Process states at the end of each episode
        # print("Rollout end")
        # np.save("states.npy", np.array(self.states))
        # self.episode_count += 1
        # if self.episode_count % self.render_every == 0:
        #     self.training_env.render()

        # self.states = []  # Reset the state tracker for the new episode
        pass

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        # print("Training is complete.")
        np.save("states.npy", np.array(self.states))
        self.training_env.render()
        self.states = []  # Clean up the states list


class VQVAE_Env(gym.Env):
    """
    Custom Gym environment for VQVAE-based state representation learning.

    The environment simulates interactions with a latent space of a neural network using a Vector Quantized Variational Autoencoder (VQVAE).
    Actions modify the latent representation, aiming to optimize the network's performance as assessed by a surrogate model.

    Parameters:
    - embed_dim: Dimension of the latent space.
    - num_embeddings: Number of embeddings in the VQVAE codebook.
    - max_allowed_actions: Maximum number of actions before the episode ends.
    - surrogate_model: A model that estimates the performance (e.g., accuracy) of the neural network based on the latent representation.
    - decoder: Model to decode the latent representation to a form that the surrogate model can evaluate.
    - codebook: The set of embeddings (codebook) learned by the VQVAE.
    - num_previous_actions: The number of previous actions to keep in the action history.

    Functions:
    - reset(seed=None): Resets the environment to an initial state.
    - step(action): Executes the given action in the environment.
    - render(): A placeholder for rendering the environment's state, if needed.
    - close(): A placeholder for any cleanup necessary to close the environment.
    - validate_action(action): Validates the given action to ensure it is within the expected range.
    - calculate_reward(): Calculates the reward based on the change in performance (e.g., accuracy) as assessed by the surrogate model.
    - sample_action(): Samples a random action from the action space.
    - get_state(): Returns the current state of the environment.
    """

    def __init__(  # TODO: Take alpha and beta as hyperparameters --> once we have the surrogate model
        self,
        embed_dim,
        num_embeddings,
        max_allowed_actions,
        surrogate_model,
        decoder,
        codebook,
        consider_previous_actions=False,
        num_previous_actions=4,
        render_mode="human",
        render_labels=None,
        render_data=None,
        log_dir="trainingLogs",
    ):
        super().__init__()

        # Environment Parameters
        self.embed_dim = embed_dim  # dimension of the codebook and state vectors
        self.num_codebook_vectors = num_embeddings  # +1 for the stop action
        self.terminal_action = (
            self.embed_dim * self.num_codebook_vectors
        )  # index of the terminal action
        self.max_allowed_actions = (
            max_allowed_actions  # maximum number of actions allowed before termination
        )
        self.surrogate_model = (
            surrogate_model  # surrogate model to be used for calculating the reward
        )
        self.decoder = decoder  # decoder to be convert state representation to that the surrogate model can understand
        self.codebook = codebook  # codebook to be used for taking actions
        self.num_previous_actions = num_previous_actions  # number of previous actions to be used for state representation
        self.consider_previous_actions = consider_previous_actions

        # Action space is a tuple (codebook_number, codebook_index)
        self.action_space = spaces.Discrete(
            self.num_codebook_vectors * self.embed_dim + 1
        )

        # Observation space is now a tuple of the latent vector and action history

        if self.consider_previous_actions:
            self.observation_space = spaces.Dict(
                {
                    "latent_vector": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.embed_dim,),
                        dtype=np.float32,
                    ),
                    "action_history": spaces.Box(
                        low=-1,
                        high=(self.num_codebook_vectors * self.embed_dim),
                        shape=(self.num_previous_actions,),
                        dtype=np.int32,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.embed_dim,), dtype=np.float32
            )

        self.render_mode = render_mode
        self.render_labels = render_labels
        self.render_data = render_data
        self.tsne = TSNE(n_components=2, random_state=42)
        self.log_dir = log_dir
        current_path = os.getcwd()
        self.log_dir = os.path.join(current_path, self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_dir = os.path.join(self.log_dir, "render")
        os.makedirs(self.log_dir, exist_ok=True)

        # Reset the environment
        self.reset()

    def step(self, action_input):

        # Check if the episode is done
        if self.is_episode_done:
            # Give a warning here to indicate the episode has ended
            print(
                "Warning: Attempted to step after episode is done. Please reset environment."
            )
            return None, None, True, False, {}

        if action_input != self.terminal_action:

            # Convert the action to a tuple
            action = (
                action_input // self.embed_dim,
                action_input % self.embed_dim,
            )

            # Check if the action is valid
            assert self.validate_action(
                action
            ), "Illegal Action - Action is out of bounds."

            # Extract the action
            codebook_number, codebook_index = action

            # Update the state with the action
            self.state[codebook_index] = self.codebook[codebook_number][
                codebook_index
            ]  # might want to index codebook as self.codebook[codebook_number, codebook_index] instead!
            if self.consider_previous_actions:
                self.action_history.append(action_input)
            self.step_count += 1

            # Calculate the reward
            reward = self.calculate_reward()

            # Check if the maximum number of actions has been reached
            done = self.step_count >= self.max_allowed_actions

            if self.consider_previous_actions:
                # Convert action_history deque to a numpy array and reshape
                action_history_array = np.array(
                    list(self.action_history), dtype=np.int32
                ).flatten()

                # Update the observation state
                obs_state = {
                    "latent_vector": self.state,
                    "action_history": action_history_array,
                }
            else:
                obs_state = self.state

            # Check if the episode is done
            if done:
                self.is_episode_done = True

            return (
                obs_state,
                reward,
                done,
                done,
                {},
            )  # Return done for truncated episodes

        else:
            # If the terminal action is taken, then the episode is done
            done = True
            reward = 0.0  # No reward for the terminal action
            # Ensure action_history is correctly formatted even at episode end
            if self.consider_previous_actions:
                action_history_array = np.array(
                    list(self.action_history), dtype=np.int32
                ).flatten()
                obs_state = {
                    "latent_vector": self.state,
                    "action_history": action_history_array,
                }
            else:
                obs_state = self.state
            self.is_episode_done = True
            return obs_state, reward, done, False, {}

    def validate_action(self, action):
        codebook_number, codebook_index = action
        return (0 <= codebook_number < self.num_codebook_vectors) and (
            0 <= codebook_index < self.embed_dim
        )

    def calculate_reward(
        self,
    ):  # We can add alpha and beta hyperparameters, for defining the linear combination
        decoded_state = self.decoder(torch.from_numpy(self.state))  # Decode the state
        if decoded_state.dim() == 1:
            decoded_state = decoded_state.unsqueeze(0)
        accuracy = self.surrogate_model.evaluate(
            decoded_state
        )  # Calculate the accuracy
        reward = accuracy - self.previous_accuracy  # Reward is the change in accuracy
        self.previous_accuracy = accuracy  # Update the previous accuracy
        return float(reward[0])  # TODO: FIXXX THISSSS!!!!!!!!

    def reset(self, seed=None):

        # Random seed initialization for reproducibility
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.state = self.np_random.normal(size=self.embed_dim).astype(np.float32)
        if self.consider_previous_actions:
            self.action_history = deque(
                [-1] * self.num_previous_actions, maxlen=self.num_previous_actions
            )
        self.step_count = 0
        self.previous_accuracy = 0.0  # Initial accuracy (assuming 0)
        self.is_episode_done = False

        if self.consider_previous_actions:
            # Convert action_history deque to a numpy array and reshape
            action_history_array = np.array(
                list(self.action_history), dtype=np.int32
            ).flatten()

            # Update the observation state
            obs_state = {
                "latent_vector": self.state,
                "action_history": action_history_array,
            }
        else:
            obs_state = self.state

        return obs_state, {}

    def sample_action(self):
        return self.action_space.sample()

    def get_state(self):
        return self.state

    def render(self, mode="human"):
        if mode == "human":
            states = np.load("states.npy")
            print(states.shape)
            states = np.squeeze(states, axis=1)
            print("Rendering the environment...")
            if self.render_data is not None and self.render_labels is not None:
                latent_vectors = np.load(self.render_data)
                labels = np.load(self.render_labels)
                codebook = self.codebook.numpy()
                print(latent_vectors.shape, codebook.shape, states.shape)
                all_vectors = np.vstack(
                    [latent_vectors, codebook, states]
                )  # Include states for TSNE
                all_vectors_2d = self.tsne.fit_transform(all_vectors)

                # Extract transformed latent vectors, codebook vectors, and states
                latent_vectors_2d = all_vectors_2d[: len(latent_vectors)]
                codebook_vectors_2d = all_vectors_2d[len(latent_vectors) : -len(states)]
                all_states_2d = all_vectors_2d[-len(states) :]
            for i in range(len(all_states_2d) // 100):
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(
                    latent_vectors_2d[:, 0],
                    latent_vectors_2d[:, 1],
                    c=labels,
                    cmap="tab10",
                    alpha=0.6,
                )
                plt.scatter(
                    codebook_vectors_2d[:, 0],
                    codebook_vectors_2d[:, 1],
                    color="black",
                    marker="x",
                )  # Codebook vectors in black
                states_2d = all_states_2d[i * 100 : i * 100 + 100]
                plt.plot(
                    states_2d[:, 0],
                    states_2d[:, 1],
                    color="red",
                    linestyle="-",
                    linewidth=2,
                    marker="o",
                    markersize=4,
                )

                # Create a legend for the labels
                unique_labels = np.unique(labels)
                colors = scatter.cmap(np.linspace(0, 1, len(unique_labels)))
                legend_handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=str(int(label)),
                        markersize=10,
                        markerfacecolor=color,
                    )
                    for label, color in zip(unique_labels, colors)
                ]
                plt.legend(handles=legend_handles, title="Labels")

                plt.title(
                    "t-SNE Visualization of Latent Vectors, Codebook Vectors, and Trajectories"
                )
                plt.xlabel("t-SNE Component 1")
                plt.ylabel("t-SNE Component 2")
                plt.grid(True)
                now = datetime.datetime.now()
                timestamp_str = now.strftime("%Y%m%d_%H%M%S")
                plt.savefig(f"{self.log_dir}/latent_space_{timestamp_str}.png")
        else:
            print("Rendering mode not supported.")
            super().render(mode=mode)

    def close(self):
        pass
