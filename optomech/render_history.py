import os
import glob
import json
import pickle
import argparse
import warnings
import imageio
import numpy as np
import re

from pathlib import Path

import hcipy
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

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

            # print(file_mtime)
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
    print("Gathering step info pickle files from %s" % episode_file)
    step_info_pickle_filenames = glob.glob(os.path.join(episode_file, "*.pkl"))
    print(f"Found {len(step_info_pickle_filenames)} step info pickle files.")

    # Add a test here ensuring some files were found.
    if len(step_info_pickle_filenames) == 0:
        raise RuntimeError("No step info pickle files found in %s" % episode_file)

    # Iterate over each pickled dict loading each into memory, keyed by step index.
    # step_info_dicts = dict()

    # for step_info_pickle_filename in step_info_pickle_filenames:
    #     with open(step_info_pickle_filename, 'rb') as f:

    #         info_dict = pickle.load(f)
    #         step_info_dicts[info_dict["step_index"]] = info_dict


    # Build a list to hold the ordered filenames of the renders for gifing.
    render_filenames = list()

    reward_values = list()

    surface_min = None
    surface_max = None
    for step_info_pickle_filename in step_info_pickle_filenames:
        with open(step_info_pickle_filename, 'rb') as f:
            a = pickle.load(f)
            if a['state_content'][0]['segmented_mirror_surfaces']:
                b = a['state_content'][0]['segmented_mirror_surfaces'][0]
                if surface_min is None or np.min(b) < surface_min:
                    surface_min = np.min(b)
                if surface_max is None or np.max(b) > surface_max:
                    surface_max = np.max(b)
    # print("max, min")
    surface_max = surface_max * 1e6
    surface_min = surface_min * 1e6
    steps_list = list()

    # Print progress information
    total_files = len(step_info_pickle_filenames)
    print(f"🎨 Rendering {total_files} step files...")

    # Now we can iterate over each steps info do as we please.
    # for i, (step_index, step_info_dict) in enumerate(sorted(step_info_dicts.items())):
    for i, (step_info_pickle_filename) in enumerate(natural_sort(step_info_pickle_filenames)):
                                                    
        with open(step_info_pickle_filename, 'rb') as f:
            info_dict = pickle.load(f)
            step_index= info_dict["step_index"] 
            step_info_dict = info_dict

        DEBUG: print("Rendering step %d" % step_index)

        # If the step index is not a multiple of the render interval, skip it.
        if step_index % flags.render_interval != 0:
            continue
        
        # Only add to steps_list if we're actually rendering this step
        steps_list.append(step_index)
    
        try:


            # Build a dir inside of the episode log dir for these renders.
            observation_path = os.path.join(renders_path, "observations")
            Path(observation_path).mkdir(parents=True, exist_ok=True)

            # If a gif directory doesn't already exist, make one.
            gif_path = os.path.join(observation_path, 'gif')
            Path(gif_path).mkdir(parents=True, exist_ok=True)

            # Parse the environment-level metadata we'll need for labeling.
            frame_interval_ms = env_metadata_dict["frame_interval_ms"]
            ao_interval_ms = env_metadata_dict["ao_interval_ms"]
            frames_per_decision = env_metadata_dict["frames_per_decision"]
            commands_per_decision = env_metadata_dict["commands_per_decision"]
            ao_steps_per_frame = env_metadata_dict["ao_steps_per_frame"]

            # Compute the start time of the step we're about to render.
            step_start_time = frame_interval_ms * frames_per_decision * step_index


            # Build the filename and full save path of this render.
            render_filename = 'agent_step_' + str(i)
            render_filename = render_filename + '.png'
            save_path = os.path.join(observation_path, render_filename)

            # Add this render's filename to the list of render filenames.
            render_filenames.append(save_path)

            plt.clf()   
                
            num_rows = 2
            num_cols = np.max([frames_per_decision, commands_per_decision * 3])
            num_cols = (2*frames_per_decision) + 1

            mag = 4
            # fig = plt.figure(figsize=(num_cols * mag, num_rows * mag))
            fig = plt.figure(figsize=(3 * mag, num_rows * mag))

            # Build the render labels.
            title = 'Agent ($t~=~%.3f~ms,~step~%d$)' % (step_start_time, step_index)
            plt.suptitle(title)

            # This loop runs once per step and there is one step per decision.
            for frame_index in range(frames_per_decision):

                # Parse a single frame from the observation...
                obs_frame = step_info_dict["observation"][frame_index]
                # DEBUG: print(step_info_dict["state_content"])
            
                plt.subplot(num_rows, num_cols, frame_index + 1)

                maxpool_obs_frame = False
                if maxpool_obs_frame:

                    M, N = obs_frame.shape
                    K = 8
                    L = 8

                    MK = M // K
                    NL = N // L
                    obs_frame = obs_frame[:MK*K, :NL*L].reshape(MK, K, NL, L).mean(axis=(1, 3))

                def radial_profile(data, center=None):
                    """
                    Compute azimuthal average (radial profile) of a 2D array.
                    """
                    y, x = np.indices(data.shape)
                    if center is None:
                        center = np.array([(x.max()-x.min()) / 2.0, (y.max()-y.min()) / 2.0])
                    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
                    r_int = r.astype(int)

                    radial_sum = np.bincount(r_int.ravel(), weights=data.ravel())
                    radial_count = np.bincount(r_int.ravel())
                    radial_profile = radial_sum / np.maximum(radial_count, 1)

                    return radial_profile


                def radial_profile_by_angle(data, center=None, num_angles=180):
                    """
                    Compute radial profiles within angular segments (azimuthal bins).
                    
                    Parameters:
                    - data: 2D ndarray (image)
                    - center: (x, y) tuple (defaults to image center)
                    - num_angles: number of angular segments (e.g. 180 for 2° resolution)

                    Returns:
                    - profiles: shape (num_angles, max_radius)
                    """
                    h, w = data.shape
                    y, x = np.indices((h, w))

                    if center is None:
                        center = (w / 2, h / 2)
                    cx, cy = center

                    dx = x - cx
                    dy = y - cy

                    r = np.sqrt(dx**2 + dy**2)
                    theta = (np.arctan2(dy, dx) + 2 * np.pi) % (2 * np.pi)

                    r_int = r.astype(np.int32).ravel()
                    theta_bin = np.floor(theta * num_angles / (2 * np.pi)).astype(np.int32).ravel()
                    theta_bin = np.clip(theta_bin, 0, num_angles - 1)

                    max_radius = r_int.max() + 1

                    # Prepare output arrays
                    profiles = np.zeros((num_angles, max_radius), dtype=np.float32)
                    counts = np.zeros((num_angles, max_radius), dtype=np.int32)

                    # Flatten image data
                    flat_data = data.ravel()

                    # Use advanced indexing for accumulation
                    np.add.at(profiles, (theta_bin, r_int), flat_data)
                    np.add.at(counts, (theta_bin, r_int), 1)

                    # Avoid division by zero
                    profiles = profiles / np.maximum(counts, 1)

                    return profiles
                def compute_fringe_quality(image: np.ndarray) -> float:
                    """
                    Compute a scalar score for circular fringe quality (NumPy-only).
                    Input image should be 2D float32 or float64 in [0, 1].
                    """
                    # Step 1: Remove DC and apply Hanning window
                    image_centered = image - np.mean(image)
                    window = np.outer(np.hanning(image.shape[0]), np.hanning(image.shape[1]))
                    windowed = image_centered * window

                    # Step 2: FFT and log-MTF
                    fft = np.fft.fftshift(np.fft.fft2(windowed))
                    log_mtf = np.log10(np.abs(fft) + 1e-6)

                    # Step 3: Radial profile
                    center = (image.shape[1] // 2, image.shape[0] // 2)
                    # radial_profile_vals = radial_profile_by_angle(
                    #     log_mtf, 
                    #     center=center,
                    #     num_angles=4)
                    radial_profile_val = radial_profile(
                        log_mtf, 
                        center=center)
                    

                    # Step 4: Peak detection
                    from scipy.signal import find_peaks

                    total_peaks = 0

                    # for radial_profile_val in radial_profile_vals:

                    #     print("radial_profile_val")
                    #     print(radial_profile_val)
                    #     peaks, props = find_peaks(radial_profile_val,
                    #                           prominence=0.2)
                    #     total_peaks += len(peaks)

                        # plt.title(len(peaks))
                        # plt.plot(radial_profile_val)
                        # plt.show()

                    
                    # print("radial_profile_val")
                    # print(radial_profile_val)
                    peaks, props = find_peaks(radial_profile_val,
                                                prominence=0.2)
                    

                    total_peaks += len(peaks)

                    # plt.title(len(peaks))
                    # plt.plot(radial_profile_val)
                    # plt.show()
                    if len(peaks) == 0:
                        return 0.0

                    # peak_heights = props['peak_heights']
                    # print("peaks")
                    # print(peak_heights)

                    # Step 5: Score = sum of peak heights × number of peaks / length of profile
                    # score = np.sum(peak_heights) * len(peaks) / len(radial_profile_vals)
                    score = total_peaks
                    return score

                plt.title('log Science Image ($o_{t-1}$)')
                plt.imshow(np.log(obs_frame), cmap='inferno',)
                # Compute the MTF of the science image
                mtf = np.log(np.abs(np.fft.fftshift(np.fft.fft2(obs_frame))))
                compute_fringe_quality(obs_frame / np.max(obs_frame))
                # plt.title(compute_fringe_quality(mtf))
                # plt.imshow(mtf, cmap='inferno')
                plt.colorbar()



                plt.subplot(num_rows, num_cols, frame_index + 2)
                plt.title('Science Image ($o_{t-1}$)')
                plt.imshow(obs_frame, cmap='inferno') #

                # Show the center 64 pixels of the image
                # use the image size to determine the center
                center = obs_frame.shape[0] // 2
                # plt.imshow(obs_frame[center-32:center+32, center-32:center+32], cmap='inferno') #
                plt.colorbar()
            

            if len(step_info_dict["action"]) != 0:
                use_flat_action = True

                if use_flat_action:

                        plt.subplot(num_rows, num_cols, num_cols)
                        plt.title('Action ($a_t$)')
                        # DEBUG: print(len(step_info_dict["action"]))
                        # DEBUG: print(step_info_dict["action"])
                        
                        plt.imshow([step_info_dict["action"] for _ in range(len(step_info_dict["action"]))], cmap='inferno')
                        # plt.imshow([step_info_dict["action"] for _ in range(len(step_info_dict["action"]))], cmap='inferno')
                        plt.colorbar()

            segmented_mirror_surface = step_info_dict['state_content'][0]['segmented_mirror_surfaces'][0]
            #['segmented_mirror_surfaces'][0]
            # print(segmented_mirror_surface)
            post_dm_wavefront = step_info_dict['state_content'][0]['post_dm_wavefronts'][0]
            
            aperture_mask = post_dm_wavefront.intensity > 0.0
            
            plt.subplot(2, 3, 4)
            plt.title('Secondary Surface [$\\mu$m]')
            # Compute the L2 norm of the segmented mirror surface
            secondary_mirror_surface_l2 = np.linalg.norm(segmented_mirror_surface)

            plt.title(secondary_mirror_surface_l2)


            # hcipy.imshow_field(segmented_mirror_surface * 1e6, cmap='RdBu', vmin=-1.1, vmax=1.1, mask=aperture_mask)
            # hcipy.imshow_field(segmented_mirror_surface * 1e6, cmap='RdBu', vmin=surface_min, vmax=surface_max, mask=aperture_mask)
            hcipy.imshow_field(segmented_mirror_surface * 1e6, cmap='RdBu', mask=aperture_mask)

            # print("+segmented_mirror_surface min and max")
            # print(np.min(segmented_mirror_surface * 1e6))
            # print(np.max(segmented_mirror_surface * 1e6))
            # print("-segmented_mirror_surface min and max")

            # hcipy.imshow_field(segmented_mirror_surface * 1e6, cmap='RdBu', mask=aperture_mask)
            # plt.imshow(obs_frame, cmap='inferno') #
            plt.colorbar()

            if step_info_dict['state_content'][0]['dm_surfaces']:
                dm_surface = step_info_dict['state_content'][0]['dm_surfaces'][0]
                plt.subplot(2, 3, 5)
                plt.title('DM Surface [$\\mu$m]')
                # hcipy.imshow_field(segmented_mirror_surface * 1e6, cmap='RdBu', vmin=-1.1, vmax=1.1, mask=aperture_mask)
                # hcipy.imshow_field(dm_surface * 1e6, cmap='RdBu', mask=aperture_mask)
                hcipy.imshow_field(dm_surface * 1e6, cmap='RdBu',mask=aperture_mask)
                # plt.imshow(obs_frame, cmap='inferno') #
                plt.colorbar()


            # DEBUG: print(step_info_dict["reward"])

            reward = step_info_dict["reward"]

            # strehl = (np.e ** (np.log(reward + 1.0)/ np.e))

            reward_values.append(reward)
            plt.subplot(2, 3, 6)
            plt.title('Reward')
            # print()
            plt.plot(steps_list, reward_values)
            # plt.ylim(bottom=-1.1, top=0.1)
            # plt.ylim(bottom=0.0, top=1.0)


            plt.subplots_adjust(wspace=0.25, hspace=0.25)
            render_dpi = flags.render_dpi
            fig.set_dpi(render_dpi)
            fig.canvas.draw()
            plt.savefig(save_path,
                        pad_inches=0.1,
                        dpi=render_dpi)
            
            plt.close()

        except Exception as e:
            print("Error rendering step %d: %s" % (step_index, e))
            continue
            
    print(f"🎬 Creating GIF from {len(render_filenames)} rendered images...")
    make_gif_from_images(gif_path=gif_path,
                            gif_name='observations.gif',
                            image_filenames=render_filenames)


def make_gif_from_images(gif_path, gif_name, image_filenames):

    # If a gif directory doesn't already exist, make one.
    Path(gif_path).mkdir(parents=True, exist_ok=True)
        
    images = list()
    for filename in natural_sort(image_filenames):

        # print(filename)
        try:
            images.append(imageio.imread(filename))
        except Exception as e:
            print("Error reading image %s: %s" % (filename, e))
            continue
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
    im = ax.imshow(np.log(image), cmap='inferno')
        
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


def render_dm_image(image, save_path, title, render_dpi=400):

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
    im = ax.imshow(np.log(image), cmap='inferno')
        
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
                        default=100,
                        help='The DPI of all rendered images.')
    
    parser.add_argument('--render_mode',
                        type=str,
                        default="simple",
                        help='The type of rendering desired.')
    
    parser.add_argument('--render_interval',
                        type=int,
                        default=1,
                        help='The interval at which to render images.')
    
    parser.add_argument('--log_scale_images',
                        type=bool,
                        default=False,
                        help='If True, log-scale science images.')
    

    
    
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)


