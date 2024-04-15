import gymnasium as gym
import numpy as np
from gymnasium import spaces
from collections import deque


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

    def __init__(
        self,
        embed_dim,
        num_embeddings,
        max_allowed_actions,
        surrogate_model,
        decoder,
        codebook,
        num_previous_actions=4,
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

        # Action space is a tuple (codebook_number, codebook_index)
        self.action_space = spaces.Discrete(
            self.num_codebook_vectors * self.embed_dim + 1
        )

        # Observation space is now a tuple of the latent vector and action history

        self.observation_space = spaces.Dict(
            {
                "latent_vector": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.embed_dim,), dtype=np.float32
                ),
                "action_history": spaces.Box(
                    low=-1,
                    high=(self.num_codebook_vectors * self.embed_dim),
                    shape=(self.num_previous_actions,),
                    dtype=np.int32,
                ),
            }
        )

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

        # Convert the action to a tuple
        action = (
            action_input // self.embed_dim,
            action_input % self.embed_dim,
        )

        # Check if the action is valid
        assert self.validate_action(action), "Illegal Action - Action is out of bounds."

        # Extract the action
        codebook_number, codebook_index = action

        if action_input != self.terminal_action:

            # Update the state with the action
            self.state[codebook_index] = self.codebook[codebook_number][
                codebook_index
            ]  # might want to index codebook as self.codebook[codebook_number, codebook_index] instead!
            self.action_history.append(action_input)
            self.step_count += 1

            # Calculate the reward
            reward = self.calculate_reward()

            # Check if the maximum number of actions has been reached
            done = self.step_count >= self.max_allowed_actions

            # Convert action_history deque to a numpy array and reshape
            action_history_array = np.array(
                list(self.action_history), dtype=np.int32
            ).flatten()

            # Update the observation state
            obs_state = {
                "latent_vector": self.state,
                "action_history": action_history_array,
            }

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
            action_history_array = np.array(
                list(self.action_history), dtype=np.int32
            ).flatten()
            obs_state = {
                "latent_vector": self.state,
                "action_history": action_history_array,
            }
            self.is_episode_done = True
            return obs_state, reward, done, False, {}

    def validate_action(self, action):
        codebook_number, codebook_index = action
        return (0 <= codebook_number < self.num_codebook_vectors) and (
            0 <= codebook_index < self.embed_dim
        )

    def calculate_reward(self):
        decoded_state = self.decoder.decode(self.state)  # Decode the state
        accuracy = self.surrogate_model.evaluate(
            decoded_state
        )  # Calculate the accuracy
        reward = accuracy - self.previous_accuracy  # Reward is the change in accuracy
        self.previous_accuracy = accuracy  # Update the previous accuracy
        return reward

    def reset(self, seed=None):

        # Random seed initialization for reproducibility
        self.np_random, seed = gym.utils.seeding.np_random(seed)

        self.state = self.np_random.normal(size=self.embed_dim).astype(np.float32)
        self.action_history = deque(
            [-1] * self.num_previous_actions, maxlen=self.num_previous_actions
        )
        self.step_count = 0
        self.previous_accuracy = 0.0  # Initial accuracy (assuming 0)
        self.is_episode_done = False

        # Convert action_history deque to a numpy array for the observation
        # Reshape to ensure it matches the expected shape (num_previous_actions, 1)
        action_history_array = np.array(
            list(self.action_history), dtype=np.int32
        ).flatten()

        # Update the observation state
        obs_state = {
            "latent_vector": self.state,
            "action_history": action_history_array,
        }

        return obs_state, {}

    def sample_action(self):
        return self.action_space.sample()

    def get_state(self):
        return self.state

    def render(self):
        pass

    def close(self):
        pass
