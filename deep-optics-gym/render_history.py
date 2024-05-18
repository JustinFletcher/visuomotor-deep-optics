import os
import glob
import json
import pickle
import argparse
import warnings
import imageio

from pathlib import Path

import hcipy
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def cli_main(flags):


    # First, get the step info pickle filenames.
    if not any("episode_metadata.json" in s for s in os.listdir(flags.episode_info_dir)):

        # If more than one episode is in the choosen dir, get only the latest with warning.
        warnings.warn("The provided episode_info_dir does not contain episode " + \
                      "metadata. Assuming it's a parent to episodes and using the " + \
                      "most recently modified child directory.", ResourceWarning)
        
        # Get the names of files in the directory that has been most recently modified.
        latest_mtime = 0.0
        latest_file = None
        for filename in os.listdir(flags.episode_info_dir):
            file_mtime = os.path.getmtime(os.path.join(flags.episode_info_dir, filename))
            if file_mtime >= latest_mtime:
                latest_file = filename

            print(file_mtime)
        episode_file = glob.glob(os.path.join(flags.episode_info_dir, latest_file))[-1]

    else:
        
        # Else, the provided dir is a single episode, so simply get it's content.
        episode_file = flags.episode_info_dir

    
    # Get metadata dict.
        
    with open(os.path.join(episode_file, "episode_metadata.json"), 'rb') as f:
    
        env_metadata_dict = json.load(f)

    # Create the save directory if it doesn't already exist.
    renders_path = os.path.join(episode_file, "renders")
    Path(renders_path).mkdir(parents=True, exist_ok=True)
    
    # Get a list of all pickle files in the episode directory.
    step_info_pickle_filename = glob.glob(os.path.join(episode_file, "*.pkl"))

    # Iterate over each pickled dict loading each into memory, keyed by step index.
    step_info_dicts = dict()

    for step_info_pickle_filename in step_info_pickle_filename:
        with open(step_info_pickle_filename, 'rb') as f:

            info_dict = pickle.load(f)
            step_info_dicts[info_dict["step_index"]] = info_dict

    # TODO: Externalize.
    render_mode = "simple"

    # Build a list to hold the ordered filenames of the renders for gifing.
    render_filenames = list()

    # Now we can iterate over each steps info do as we please.
    for step_index, step_info_dict in sorted(step_info_dicts.items()):

        if render_mode == "simple":

            # Build a dir inside of the episode log dir for these renders.
            observation_path = os.path.join(renders_path, "observations")
            Path(observation_path).mkdir(parents=True, exist_ok=True)

            # If a gif directory doesn't already exist, make one.
            gif_path = os.path.join(observation_path, 'gif')
            Path(gif_path).mkdir(parents=True, exist_ok=True)
        
            # Parse the environment-level metadata we'll need for labeling.
            frame_interval_ms = env_metadata_dict["frame_interval_ms"]
            frames_per_decision = env_metadata_dict["frames_per_decision"]

            # Compute the start time of the step we're about to render.
            step_start_time = frame_interval_ms * frames_per_decision * step_index

            for observation_slice_index in range(frames_per_decision):

                # Parse a single frame from the observation...
                obs_frame = step_info_dict["observation"][observation_slice_index]

                # ...and compute the episode time at which it was taken.
                frame_time = step_start_time + observation_slice_index * frame_interval_ms

                # Build the filename and full save path of this render.
                render_filename = 'observation_step_' + str(step_index)
                render_filename = render_filename + '_ob_' + str(observation_slice_index) + '.png'
                save_path = os.path.join(observation_path, render_filename)

                # Add this render's filename to the list of render filenames.
                render_filenames.append(save_path)

                # Build the render labels.
                title = 'Observation ($t =%.3f, step %d$)' % (frame_time, step_index)

                # Render the image, saving it in the provided path.
                render_standard_image(image=obs_frame,
                                      save_path=save_path,
                                      title=title,
                                      render_dpi=flags.render_dpi)
            
    make_gif_from_images(gif_path=gif_path,
                            gif_name='observations.gif',
                            image_filenames=render_filenames)


def make_gif_from_images(gif_path, gif_name, image_filenames):

    # If a gif directory doesn't already exist, make one.
    Path(gif_path).mkdir(parents=True, exist_ok=True)
        
    images = list()
    for filename in image_filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(gif_path, gif_name), images)

    return

def render_standard_image(image, save_path, title, render_dpi=400):

    num_rows = 1
    num_cols = 1

    mag = 4
    fig = plt.figure(figsize=(num_cols * mag, num_rows * mag))

    ax = plt.subplot2grid(
                (num_rows, num_cols),
                (0, 0),
                colspan=1,
                rowspan=1)
    
    ax.set_title(title)
    im = ax.imshow(image, cmap='inferno')
        
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.tight_layout()

    plt.axis('off')


    fig.set_dpi(render_dpi)
    fig.canvas.draw()
    plt.savefig(save_path,
                pad_inches=0.1,
                dpi=render_dpi)
    
    plt.close()

    return

if __name__ == "__main__":


    # Instantiate an arg parser
    parser = argparse.ArgumentParser()

    # Set arguments and their default values
    parser.add_argument('--episode_info_dir',
                        type=str,
                        default="./tmp/",
                        help='The directory from which to read state data.')
    
    parser.add_argument('--render_dpi',
                        type=int,
                        default=400,
                        help='The DPI of all rendered images.')
    

    
    
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)


