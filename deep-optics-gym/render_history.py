import os
import glob
import json
import pickle
import argparse
import warnings
import imageio
import numpy as np

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


    # Build a list to hold the ordered filenames of the renders for gifing.
    render_filenames = list()



    reward_values = list()

    # Now we can iterate over each steps info do as we please.
    for step_index, step_info_dict in sorted(step_info_dicts.items()):

        print("Rendering step %d" % step_index)

        if flags.render_mode == "simple":

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
                frame_time = step_start_time + (observation_slice_index * frame_interval_ms)

                # Build the filename and full save path of this render.
                render_filename = 'observation_step_' + str(step_index)
                render_filename = render_filename + '_ob_' + str(observation_slice_index) + '.png'
                save_path = os.path.join(observation_path, render_filename)

                # Add this render's filename to the list of render filenames.
                render_filenames.append(save_path)

                # Build the render labels.
                title = 'Observation ($t = %.3f ms, step %d$)' % (frame_time, step_index)

                # Render the image, saving it in the provided path.
                render_standard_image(image=obs_frame,
                                      save_path=save_path,
                                      title=title,
                                      render_dpi=flags.render_dpi)
                
        elif flags.render_mode == "dm":

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
            ao_steps_per_frame = env_metadata_dict["ao_steps_per_frame"]

            # Compute the start time of the step we're about to render.
            step_start_time = frame_interval_ms * frames_per_decision * step_index

            # This loop runs once per step and there is one step per decision.
            for frame_index in range(frames_per_decision):

                # Parse a single frame from the observation...
                obs_frame = step_info_dict["observation"][frame_index]

                for ao_step_per_frame_index in range(ao_steps_per_frame):


                    ao_step_index = (ao_steps_per_frame * frame_index) + ao_step_per_frame_index

                    print(ao_step_index)
                    # Parse a single frame from the observation...
                    dm_surface = step_info_dict['state_content']['dm_surfaces'][ao_step_index]
                    atmos_layer = step_info_dict['state_content']['atmos_layer_0_list'][ao_step_index]
                    instantaneous_psf = step_info_dict['state_content']['instantaneous_psf'][ao_step_index]
                    post_dm_wavefront = step_info_dict['state_content']['post_dm_wavefronts'][ao_step_index]
                    segmented_mirror_surface = step_info_dict['state_content']['segmented_mirror_surfaces'][ao_step_index]
                    pre_atmosphere_object_wavefront = step_info_dict['state_content']['pre_atmosphere_object_wavefronts'][ao_step_index]
                    post_atmosphere_wavefront = step_info_dict['state_content']['post_atmosphere_wavefronts'][ao_step_index]
                    
                    wavelength = step_info_dict['state_content']['wavelength']

                    print("len %d" % len(step_info_dict['state_content']['instantaneous_psf']))

                    # TODO: Refactor both of these using an episode-level state storage option.
                    aperture = step_info_dict["aperture"][frame_index]
                    wavelength = step_info_dict['state_content']['wavelength']
                    aperture_mask = post_dm_wavefront.intensity > 0.0

                    print(aperture_mask)


                    # ...and compute the episode time at which it was taken.
                    frame_time = step_start_time + (ao_step_index * ao_interval_ms)

                    # Build the filename and full save path of this render.
                    render_filename = 'dm_step_' + str(step_index)
                    render_filename = render_filename + '_ao_' + str(ao_step_index) + '.png'
                    save_path = os.path.join(observation_path, render_filename)

                    # Add this render's filename to the list of render filenames.
                    render_filenames.append(save_path)

                    # Build the render labels.
                    title = 'DM ($t = %.3f ms, step %d$)' % (frame_time, step_index)

                    plt.clf()
                    
                    num_rows = 2
                    num_cols = 3

                    mag = 4
                    fig = plt.figure(figsize=(num_cols * mag, num_rows * mag))

                    plt.suptitle(title)

                    plt.subplot(num_rows,num_cols,1)
                    plt.title('DM surface [$\\mu$m]')
                    hcipy.imshow_field(dm_surface * 1e6, cmap='RdBu', vmin=-2, vmax=2, mask=aperture_mask)
                    # hcipy.imshow_field(segmented_mirror_surface * 1e6, cmap='RdBu', vmin=-7, vmax=7, mask=aperture_mask)
                    # hcipy.imshow_field(dm_surface * 1e6, cmap='RdBu')
                    plt.colorbar()

                    if atmos_layer:
                        phase_screen_phase = atmos_layer.phase_for(wavelength) # in radians
                        phase_screen_opd = phase_screen_phase * (wavelength / (2 * np.pi)) * 1e6

                        plt.subplot(num_rows,num_cols,2)
                        hcipy.imshow_field(phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')
                        plt.title('Turbulent wavefront [$\\mu$m]')
                        plt.colorbar()

                    plt.subplot(num_rows,num_cols,3)
                    plt.title('Instantaneous PSF at 2.2$\\mu$m [log]')

                    print("self.instantaneous_psf %3.16f" % np.std(instantaneous_psf))
                    plt.imshow(np.log10(instantaneous_psf/ instantaneous_psf.max()), vmin=-8, vmax=0, cmap='inferno') #
                    plt.colorbar()

                 
                    plt.subplot(num_rows,num_cols,4)
                    plt.title('Science Image')
                    plt.imshow(obs_frame, cmap='inferno') #
                    plt.colorbar()
                    
                    plt.subplot(num_rows,num_cols,5)
                    plt.title('post_dm_wavefront.phase')
                    hcipy.imshow_field(post_dm_wavefront.phase, cmap='RdBu', mask=aperture_mask)
                    plt.colorbar()
                    
                    
                    plt.subplot(num_rows,num_cols,6)
                    plt.title('post_atmosphere_wavefront.phase')
                    hcipy.imshow_field(post_atmosphere_wavefront.phase, cmap='RdBu', mask=aperture_mask)
                    # plt.imshow(obs_frame, cmap='inferno') #
                    plt.colorbar()
                    


                    render_dpi = 400
                    fig.set_dpi(render_dpi)
                    fig.canvas.draw()
                    plt.savefig(save_path,
                                pad_inches=0.1,
                                dpi=render_dpi)
                    
                    plt.close()

        elif flags.render_mode == "diffmotion":

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
            ao_steps_per_frame = env_metadata_dict["ao_steps_per_frame"]

            # Compute the start time of the step we're about to render.
            step_start_time = frame_interval_ms * frames_per_decision * step_index

            # This loop runs once per step and there is one step per decision.
            for frame_index in range(frames_per_decision):

                # Parse a single frame from the observation...
                obs_frame = step_info_dict["observation"][frame_index]

                for ao_step_per_frame_index in range(ao_steps_per_frame):


                    ao_step_index = (ao_steps_per_frame * frame_index) + ao_step_per_frame_index

                    print(ao_step_index)
                    # Parse a single frame from the observation...
                    dm_surface = step_info_dict['state_content']['dm_surfaces'][ao_step_index]
                    atmos_layer = step_info_dict['state_content']['atmos_layer_0_list'][ao_step_index]
                    instantaneous_psf = step_info_dict['state_content']['instantaneous_psf'][ao_step_index]
                    post_dm_wavefront = step_info_dict['state_content']['post_dm_wavefronts'][ao_step_index]
                    segmented_mirror_surface = step_info_dict['state_content']['segmented_mirror_surfaces'][ao_step_index]
                    pre_atmosphere_object_wavefront = step_info_dict['state_content']['pre_atmosphere_object_wavefronts'][ao_step_index]
                    post_atmosphere_wavefront = step_info_dict['state_content']['post_atmosphere_wavefronts'][ao_step_index]
                    
                    wavelength = step_info_dict['state_content']['wavelength']

                    print("len %d" % len(step_info_dict['state_content']['instantaneous_psf']))

                    # TODO: Refactor both of these using an episode-level state storage option.
                    aperture = step_info_dict["aperture"][frame_index]
                    wavelength = step_info_dict['state_content']['wavelength']
                    aperture_mask = post_dm_wavefront.intensity > 0.0

                    print(aperture_mask)


                    # ...and compute the episode time at which it was taken.
                    frame_time = step_start_time + (ao_step_index * ao_interval_ms)

                    # Build the filename and full save path of this render.
                    render_filename = 'dm_step_' + str(step_index)
                    render_filename = render_filename + '_ao_' + str(ao_step_index) + '.png'
                    save_path = os.path.join(observation_path, render_filename)

                    # Add this render's filename to the list of render filenames.
                    render_filenames.append(save_path)

                    # Build the render labels.
                    title = 'DM ($t = %.3f ms, step %d$)' % (frame_time, step_index)

                    plt.clf()
                    
                    num_rows = 2
                    num_cols = 4

                    mag = 4
                    fig = plt.figure(figsize=(num_cols * mag, num_rows * mag))

                    plt.suptitle(title)

                    plt.subplot(num_rows,num_cols,1)
                    plt.title('DM surface [$\\mu$m]')
                    hcipy.imshow_field(dm_surface * 1e6, cmap='RdBu', vmin=-2, vmax=2, mask=aperture_mask)
                    # hcipy.imshow_field(segmented_mirror_surface * 1e6, cmap='RdBu', vmin=-7, vmax=7, mask=aperture_mask)
                    # hcipy.imshow_field(dm_surface * 1e6, cmap='RdBu')
                    plt.colorbar()

                    if atmos_layer:
                        phase_screen_phase = atmos_layer.phase_for(wavelength) # in radians
                        phase_screen_opd = phase_screen_phase * (wavelength / (2 * np.pi)) * 1e6

                        plt.subplot(num_rows,num_cols,2)
                        hcipy.imshow_field(phase_screen_opd, vmin=-6, vmax=6, cmap='RdBu')
                        plt.title('Turbulent wavefront [$\\mu$m]')
                        plt.colorbar()

                    plt.subplot(num_rows,num_cols,3)
                    plt.title('Instantaneous PSF at 2.2$\\mu$m [log]')

                    print("self.instantaneous_psf %3.16f" % np.std(instantaneous_psf))
                    plt.imshow(np.log10(instantaneous_psf/ instantaneous_psf.max()), vmin=-8, vmax=0, cmap='inferno') #
                    plt.colorbar()

                 
                    plt.subplot(num_rows,num_cols,4)
                    plt.title('Science Image ')
                    plt.imshow(obs_frame, cmap='inferno') #
                    plt.colorbar()
                    
                    plt.subplot(num_rows,num_cols,5)
                    plt.title('post_dm_wavefront.phase [rad]')
                    hcipy.imshow_field(post_dm_wavefront.phase, cmap='RdBu', mask=aperture_mask)
                    plt.colorbar()
                    
                    
                    plt.subplot(num_rows,num_cols,6)
                    plt.title('post_atmosphere_wavefront.phase [rad]')
                    hcipy.imshow_field(post_atmosphere_wavefront.phase, cmap='RdBu', mask=aperture_mask)
                    # plt.imshow(obs_frame, cmap='inferno') #
                    plt.colorbar()
                    
                    plt.subplot(num_rows,num_cols,7)
                    plt.title('segmented_mirror_surface limited [$\\mu$m]')
                    hcipy.imshow_field(segmented_mirror_surface * 1e6, cmap='RdBu', vmin=-7, vmax=7, mask=aperture_mask)
                    # plt.imshow(obs_frame, cmap='inferno') #
                    plt.colorbar()
                    
                    plt.subplot(num_rows,num_cols,8)
                    plt.title('segmented_mirror_surface unlimited [$\\mu$m]')
                    hcipy.imshow_field(segmented_mirror_surface * 1e6, cmap='RdBu', mask=aperture_mask)
                    # plt.imshow(obs_frame, cmap='inferno') #
                    plt.colorbar()
                    


                    render_dpi = 400
                    fig.set_dpi(render_dpi)
                    fig.canvas.draw()
                    plt.savefig(save_path,
                                pad_inches=0.1,
                                dpi=render_dpi)
                    
                    plt.close()

        elif flags.render_mode == "agent_view":

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
            render_filename = 'agent_step_' + str(step_index)
            render_filename = render_filename + '.png'
            save_path = os.path.join(observation_path, render_filename)

            # Add this render's filename to the list of render filenames.
            render_filenames.append(save_path)

            plt.clf()   
                
            num_rows = 3
            num_cols = np.max([frames_per_decision, commands_per_decision * 2])

            mag = 4
            fig = plt.figure(figsize=(num_cols * mag, num_rows * mag))

            # Build the render labels.
            title = 'Agent ($t = %.3f ms, step %d$)' % (step_start_time, step_index)
            plt.suptitle(title)

            # This loop runs once per step and there is one step per decision.
            for frame_index in range(frames_per_decision):

                # Parse a single frame from the observation...
                obs_frame = step_info_dict["observation"][frame_index]

            
                plt.subplot(num_rows, num_cols, frame_index + 1)
                plt.title('Science Image ')
                plt.imshow(obs_frame, cmap='inferno') #
                plt.colorbar()
                


            # This loop runs once per step and there is one step per decision.
            for action_index in range(commands_per_decision):

                # Parse a single frame from the observation...
                actions = step_info_dict["action"][action_index]

                tensioner_commands = actions[1]
                tensioner_commands = np.expand_dims(tensioner_commands, axis=0)
                tensioner_commands = np.array(tensioner_commands).transpose()

                secondaries_commands = actions[0]
                secondaries_commands = np.array(secondaries_commands)

                plt.subplot(num_rows, num_cols, num_cols + 1 + (2 * action_index))
                plt.title('Tensioner Commands')
                plt.imshow(tensioner_commands, cmap='inferno')
                plt.colorbar()
                
                plt.subplot(num_rows, num_cols, num_cols + 2 + (2 * action_index))
                plt.title('Secondaries Commands')
                plt.imshow(secondaries_commands, cmap='inferno')
                plt.colorbar()
            

            reward_values.append(step_info_dict["reward"].to_dict()['values'])

            plt.subplot(num_rows, num_cols, (num_cols * 2) + 1)
            plt.title('Reward')
            plt.plot(reward_values)


            render_dpi = 400
            fig.set_dpi(render_dpi)
            fig.canvas.draw()
            plt.savefig(save_path,
                        pad_inches=0.1,
                        dpi=render_dpi)
            
            plt.close()


            
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
                        default=400,
                        help='The DPI of all rendered images.')
    
    parser.add_argument('--render_mode',
                        type=str,
                        default="simple",
                        help='The type of rendering desired.')
    

    
    
    parsed_flags, _ = parser.parse_known_args()

    # Call main.
    cli_main(parsed_flags)


