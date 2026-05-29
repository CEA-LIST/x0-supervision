import utils.config as config
from uni_controlnet.hack import disable_verbosity, enable_sliced_attention


disable_verbosity()

if config.save_memory:
    enable_sliced_attention()
