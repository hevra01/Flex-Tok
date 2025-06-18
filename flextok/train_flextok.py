from pathlib import Path
from flextok.flextok_wrapper import FlexTok
import yaml
from vae_wrapper import StableDiffusionVAE  
from model.utils.wrappers import SequentialModuleDictWrapper
from flextok.create_encoder import encoder

# Load config from file for VAE
# Always resolves the config path relative to the file you're in (not the shell working directory),
# config_path = Path(__file__).parent / "configs" / "vae.yaml"
# with open(config_path, "r") as f:
#     config = yaml.safe_load(f)

# # Extract the vae config dictionary
# vae_kwargs = config["vae_config"]

# # Initialize VAE
# vae = StableDiffusionVAE(**vae_kwargs)

# # Load config from file for Encoder
# config_path = Path(__file__).parent / "configs" / "encoder.yaml"
# with open(config_path, "r") as f:
#     config = yaml.safe_load(f)

# Initialize Encoder
# the encoder is a SequentialModuleDictWrapper which is already imported

print(encoder)
exit()
# 1. Model setup
model = FlexTok(
    vae=...,    encoder=...,
    decoder=...,
    regularizer=...,
    flow_matching_noise_module=...,
    pipeline=None,  # Not needed for training
)
model = model.to(device)