"""
This script simulates an optomechanical system using gymnasium.

Author: Justin Fletcher
Date: 3 June 2023
"""

import os
import time
import argparse

import hcipy
import numpy as np
import gymnasium as gym

from matplotlib import pyplot as plt

def cli_main(flags):

    # Set the GPUs we want the script to use/see
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_list

    # Register our custom DASIE environment.
    gym.envs.registration.register(
        id='DASIE-v1',
        entry_point='deep-optics-gym.dasie:DasieEnv',
        max_episode_steps=flags.max_episode_steps,
        reward_threshold=flags.reward_threshold,
    )

    # TODO: Externalize.
    render_mode='rgb_array'
    # Build a gym environment; pass the CLI flags to the constructor as kwargs.
    env = gym.make('DASIE-v1',
                   **vars(flags),
                   render_mode=render_mode,)
    
    env.metadata['render_modes'] = ['human', 'rgb_array']

    # Iterate over the number of desired episodes.
    for i_episode in range(flags.num_episodes):

        # Reset the environment...
        observation = env.reset()

        # ..and iterate over the desired number of steps. In each step...
        for t in range(flags.num_steps):

            if not flags.silence:

                print("Running step %s." % str(t))

            # ...render the environment...
            if flags.render:

                if not flags.record_env_state_info:
                    raise ValueError("You're trying to render, but haven't recorded the " +
                                     "step state information. Add --record_env_state_info.")

                print("Rendering step %s." % str(t))

                if render_mode == 'rgb_array':
                    rgb_array = env.render()

                    # print(rgb_array)
                    plt.imshow(rgb_array)
                    # plt.show()

                    plt.axis('off')
                    plt.savefig('debug_output.png',
                                bbox_inches='tight',
                                pad_inches=0,
                                dpi=flags.render_dpi)
                    
                    plt.close()

            # ...get an action, either random or all zeros...
            if flags.action_type == "random":

                action = env.action_space.sample()

            elif flags.action_type == "none":
                
                action = np.zeros_like(env.action_space.sample())

            else: 

                raise ValueError("Invalid action type: %s" % flags.action_type)
            
            # Your agent goes here:
            # action = agent(observation)

            # ...take that action, and parse the state.
            observation, reward, terminated, truncated, info = env.step(action)


            if flags.write_env_state_info:

                if not flags.record_env_state_info:
                    raise ValueError("You're trying to write, but haven't recorded, the " +
                                     "step state information. Add --record_env_state_info.")
                print(flags.state_info_save_dir)

                info["reward"] = reward
                info["terminated"] = terminated
                info["truncated"] = truncated
                info["action"] = action
                info["observation"] = observation
                print(info)
                # TODO: If it doesn't exist, create the disk location.
                # TODO: Create an episode dir in the chosen disk location.
                # TODO: Save the info dict there along with the action.


            # If the environment says we're done, stop this episode.
            if terminated or truncated:
                print("Episode finished after {} timesteps".format(t + 1))
                break

    # Once all episodes are complete, close the environment.
    env.close()

if __name__ == "__main__":

    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--gpu_list',
                        type=str,
                        default="0",
                        help='GPUs to use with this model.')
    
    parser.add_argument('--render',
                        action='store_true',
                        default=False,
                        help='Render the environment.')

    parser.add_argument('--report_time',
                        action='store_true',
                        default=False,
                        help='If provided, report time to run each step.')    
    
    parser.add_argument('--action_type',
                        type=str,
                        default="none",
                        help='Type of action to take ("random" or "none")')
    
    ### Gym simulation setup ###

    parser.add_argument('--object_type',
                        type=str,
                        default="binary",
                        help='Type of object to simulate.')
    
    parser.add_argument('--aperture_type',
                        type=str,
                        default="circular",
                        help='Type of aperture to simulate.')
    
    parser.add_argument('--max_episode_steps',
                        type=int,
                        default=10000,
                        help='Steps per episode limit.')

    parser.add_argument('--num_episodes',
                        type=int,
                        default=1,
                        help='Number of episodes to run.')
    
    parser.add_argument('--num_atmosphere_layers',
                        type=int,
                        default=0,
                        help='Number of atmosphere layers to simulate.')

    parser.add_argument('--reward_threshold', 
                        type=float,
                        default=25.0,
                        help='Max reward per episode.')

    parser.add_argument('--num_steps',
                        type=int,
                        default=500,
                        help='Number of steps to run.')

    parser.add_argument('--silence', action='store_true',
                        default=False,
                        help='If provided, be quiet.')

    parser.add_argument('--dasie_version', 
                        type=str,
                        default="test",
                        help='Which version of the DASIE sim do we use?')

    parser.add_argument('--render_frequency',
                        type=int,
                        default=1,
                        help='Render gif this frequently, in steps.')

    parser.add_argument('--ao_interval_ms',
                        type=float,
                        default=1.0,
                        help='Reciprocal of AO frequency in milliseconds.')
    
    parser.add_argument('--control_interval_ms',
                        type=float,
                        default=4.0,
                        help='Action control interval in milliseconds.')
    
    parser.add_argument('--frame_interval_ms',
                        type=float,
                        default=12.0,
                        help='Frame integration interval in milliseconds.')
    
    parser.add_argument('--decision_interval_ms',
                        type=float,
                        default=24.0,
                        help='Decision (inference) interval in milliseconds.')
    
    parser.add_argument('--focal_plane_image_size_pixels',
                        type=int,
                        default=256,
                        help='Size of the focal plane image in pixels.')
    
    parser.add_argument('--render_dpi',
                        type=float,
                        default=500.0,
                        help='DPI of the rendered image.')

    parser.add_argument('--record_env_state_info', action='store_true',
                        default=False,
                        help='If provided, record the environment state info.')
    
    parser.add_argument('--write_env_state_info', action='store_true',
                        default=False,
                        help='If provided, write the env state info to disk.')
    
    parser.add_argument('--state_info_save_dir',
                        type=str,
                        default="./tmp/",
                        help='The directory in which to write state data.')
    

    ############################ DASIE FLAGS ##################################
    parser.add_argument('--extended_object_image_file', type=str,
                        default=".\\resources\\sample_image.png",
                        help='Filename of image to convolve PSF with (if none, PSF returned)')

    parser.add_argument('--extended_object_distance', type=str,
                        default=None,
                        help='Distance in meters to the extended object.')

    parser.add_argument('--extended_object_extent', type=str,
                        default=None,
                        help='Extent in meters of the extended object image.')

    parser.add_argument('--observation_window_size',
                        type=int,
                        default=2**1,
                        help='Number of frames input to the model.')

    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)


