from env.surrogate.SurrogateModel import SurrogateModel
from env.vqvae.decoder import Decoder
from env.environment import VQVAE_Env, RenderCallback
from agent.RLTrainer import Trainer
import numpy as np

surrogate_model = 'env/models/surrogate_model.json'
codebook = 'env/models/codebook.pth'

def get_accuracy_difference(quantized_z, true_accuracy):
    """Returns the difference between the true accuracy and the quantized accuracy."""

    # Set up Config Variables
    decoder_config = {
    "out_dim": 22,           # Output dimension
    "embed_dim": 8,          # Embedding dimension
    "h_nodes": 512,          # Number of hidden nodes
    "dropout": 0.2,          # Dropout rate
    "scale": 2,              # Scale factor
    "num_layers": 5,         # Number of layers
    "load_path": 'env/models/decoder_model.pth', # Path to load model weights
    }

    env_config = {
        "embed_dim": decoder_config['embed_dim'],    # Embedding dimension
        "num_embeddings": 14,           # Number of embeddings
        "max_allowed_actions": 200,      # Maximum allowed actions
        "consider_previous_actions": True, # Consider previous actions
        "num_previous_actions": 6,       # Number of previous actions to consider  
        "render_mode": 'human',          # Render mode
        "render_data": 'env/render/architectures_trained_on.npy',  # Data for rendering
        "render_labels": 'env/render/labels.npy',   # Labels for rendering
        "render_log_dir": 'trainingLogs',                  # Directory for logging data
        "consider_max_params": False,   # Consider maximum parameters
        "max_params": 1e9,             # Maximum parameters
        "min_params" : 1e9,
        #"min_params" : 1e8,                # Minimum parameters
    }

    model_config = {                #TODO: Consider adding entropy coefficient as parameter and policy & value function structure parameters
        "model": "PPO",                # Model type ('PPO', 'A2C', 'DQN', etc.)
        "policy": 'MultiInputPolicy',          # Policy type
        "total_timesteps": 1000000,       # Total number of timesteps
        "verbose": 0,                  # Verbosity level
        "tensorboard_log": env_config['render_log_dir'],  # Tensorboard log directory
        "n_steps": 2048,               # Number of steps to run for each environment per update
        "progress_bar": False,          # Whether to display a progress bar
        "n_epochs": 12,                # Number of epochs
        "batch_size": 32,              # Batch size
    }

    log_config = {
        "project": 'Test',                          # Project name in wandb
        #"entity": 'trex-ai',                            # Entity name in wandb
        "sync_tensorboard": True,                           # Whether to sync TensorBoard
        "save_code": True,                                  # Whether to save code in wandb
        "model_save_path": env_config['render_log_dir'],    # Path to save the model
        "gradient_save_freq": 100,                          # Frequency to save gradients
        "verbose": 2,                                       # Verbosity level
    }

    custom_callback_function = RenderCallback()

    trainer = Trainer(surrogate_path=surrogate_model, 
                  codebook_path=codebook, 
                  decoder_config=decoder_config, 
                  env_config=env_config, 
                  model_config=model_config, 
                  log_config=log_config)
    
    decoded_z = trainer.decode_z(quantized_z)

    # Calculate the accuracy of the decoded architecture
    accuracy_z = trainer.calculate_accuracy_for_decoded_state(decoded_z)[0]
    accuracy_difference = true_accuracy - accuracy_z

    return accuracy_difference
