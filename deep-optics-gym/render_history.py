import os
import time
import argparse

import hcipy
import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt

def cli_main(flags):

    # TODO; Implement.
    raise NotImplementedError("This feature is not built yet.")


if __name__ == "__main__":

    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--state_info_read_dir',
                        type=str,
                        default="./tmp/",
                        help='The directory from which to read state data.')
    
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)


