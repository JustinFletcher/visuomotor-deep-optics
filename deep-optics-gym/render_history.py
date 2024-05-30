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
                    hcipy.imshow_field(dm_surface * 1e6, cmap='RdBu', vmin=-1, vmax=1, mask=aperture_mask)
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





    # def render(self):

    #     if self.render_mode == 'rgb_array':

    #         num_states = len(self.state)
            
    #         num_rows = 1
    #         num_cols = 2

    #         mag = 4
    #         fig = plt.figure(figsize=(num_cols * mag, num_rows * mag))

    #         ax = plt.subplot2grid(
    #                     (num_rows, num_cols),
    #                     (0, 0),
    #                     colspan=1,
    #                     rowspan=1)

    #         ax.set_title('Current State (Frame %s)' % 0)
    #         # im = ax.imshow(np.log(self.state[0] + 1e-16),
    #         #                 cmap='inferno',
    #         #                 vmin=-22.0)
    #         im = ax.imshow(self.state[0],
    #                         cmap='inferno')
    #         fig.colorbar(im, ax=ax)

                
    #         ax = plt.subplot2grid(
    #                     (num_rows, num_cols),
    #                     (0, 1),
    #                     colspan=1,
    #                     rowspan=1)
    #         ax.set_title('Log Partial PSF')
    #         im = hcipy.imshow_field(
    #             np.log((self.state_content["focal_plane_wavefronts"][0]).intensity),
    #             # np.log(self.state_content["focal_plane_wavefronts"][action_index].electric_field),
    #             cmap='gray',
    #             ax=ax)
    #         fig.colorbar(im, ax=ax)

    #         plt.tight_layout()
    #         fig.set_dpi(self.render_dpi)
    #         fig.canvas.draw()

    #         rgb_image = np.array(fig.canvas.renderer.buffer_rgba())
    #         plt.close()
    #         return rgb_image


    #     if self.render_mode == 'rgb_array_verbose':

    #         num_rows = 14
    #         num_cols = 6

    #         mag = 4
    #         fig = plt.figure(figsize=(num_cols * mag, num_rows * mag))

    #         action_list = [np.array(a) for a in self.action]
    #         num_actions = len(action_list)
    #         num_states = len(self.state)
    #         num_actions_per_state = num_actions // num_states


    #         # Build row zero, which is the state images.
    #         for state_index, state in enumerate(self.state):

    #             col_num = (num_actions_per_state * state_index)

    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (0, col_num),
    #                                   colspan=num_actions_per_state,
    #                                   rowspan=num_actions_per_state)

    #             ax.set_title('Log Current State (Frame %s)' % state_index)
    #             im = ax.imshow((state + 1e-16),
    #                            cmap='inferno',
    #                            vmin=-6)
    #             fig.colorbar(im, ax=ax)

    #         for action_index, action in enumerate(action_list):

    #             # Parse common annotations for this action.
    #             action_time = self.state_content["action_times"][action_index]
                
    #             ###############################################################
    #             # Plot a single frame read
    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (3, action_index),
    #                                   colspan=1,
    #                                   rowspan=1)
    #             ax.set_title('Log Partial Frame Read [counts] ($t$ = %.1f ms)' % action_time)
    #             im = plt.imshow(
    #                 np.log(self.state_content["readout_images"][action_index]),
    #                 cmap='gray',
    #             )
    #             fig.colorbar(im, ax=ax)
    #             ###############################################################
    #             # Plot the at-focal-plane (post-propagation) illuminance.
    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (4, action_index),
    #                                   colspan=1,
    #                                   rowspan=1)
    #             magnifier = hcipy.Magnifier(100)
 
    #             ax.set_title('Log Focal Plane Intensity [???] ($t$ = %.1f ms)' % action_time)
    #             im = hcipy.imshow_field(
    #                 np.log((self.state_content["focal_plane_wavefronts"][action_index]).intensity),
    #                 # np.log(self.state_content["focal_plane_wavefronts"][action_index].electric_field),
    #                 cmap='gray',
    #                 ax=ax)
    #             fig.colorbar(im, ax=ax)
    #             ###############################################################
    #             # Plot the (post-DM) wavefront.
    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                     (5, action_index),
    #                     colspan=1,
    #                     rowspan=1)
        
    #             ax.set_title('Post-DM Field [???] ($t$ = %.1f ms)' % action_time)
    #             im = hcipy.imshow_field(
    #                 self.state_content["post_dm_wavefronts"][action_index].intensity,
    #                 # self.state_content["post_dm_wavefronts"][action_index].electric_field,
    #                 cmap='gray',
    #                 ax=ax)
    #             fig.colorbar(im, ax=ax)

    #             ###############################################################
    #             # Plot DM surface OPD.
    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (6, action_index),
    #                                   colspan=1,
    #                                   rowspan=1)
    #             ax.set_title('DM Surface OPD [$\\mu$m] ($t$ = %.1f ms)' % action_time)
    #             im = hcipy.imshow_field(
    #                 self.state_content["dm_surfaces"][action_index] / 1e-6,
    #                 mask=self.optical_system.aperture,
    #                 cmap='RdBu_r',
    #                 ax=ax)
    #             fig.colorbar(im, ax=ax)
                
    #             ###############################################################  
    #             # Plot the command grid.
    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (7, action_index),
    #                                   colspan=1,
    #                                   rowspan=1)
                
    #             ax.set_title('Command Grid [counts] ($t$ = %.1f ms)' % action_time)
    #             im = ax.imshow(action, cmap='inferno')
    #             fig.colorbar(im, ax=ax)


    #             ###############################################################
    #             # Plot m2 at-aperture illuminance.

    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (8, action_index),
    #                                   colspan=1,
    #                                   rowspan=1)
                
    #             ax.set_title('Post-Pupil Intensity [???] ($t$ = %.1f ms)' % action_time)
    #             im = hcipy.imshow_field(
    #                 self.state_content["pupil_wavefronts"][action_index].intensity,
    #                 # self.state_content["pupil_wavefronts"][action_index].electric_field,
    #                 cmap='gray',
    #                 ax=ax)
    #             fig.colorbar(im, ax=ax)
    #             ###############################################################
    #             # Plot the segmented mirror OPD.

    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (9, action_index),
    #                                   colspan=1,
    #                                   rowspan=1)
                
                
    #             ax.set_title('Pupil OPD [$\\mu$m] ($t$ = %.1f ms)' % action_time)
    #             im = hcipy.imshow_field(
    #                 self.state_content["segmented_mirror_surfaces"][action_index] / 1e-6,
    #                 mask=self.optical_system.aperture,
    #                 cmap='RdBu_r',
    #                 ax=ax)
    #             fig.colorbar(im, ax=ax)

    #             ###############################################################
    #             # Plot the M1 at-aperture illuminance (post-atmopshere).
    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (10, action_index),
    #                                   colspan=1,
    #                                   rowspan=1)
                
    #             ax.set_title('Pre-Pupil Intensity [???] ($t$ = %.1f ms)' % action_time)
    #             im = hcipy.imshow_field(
    #                 self.state_content["post_atmosphere_wavefronts"][action_index].intensity,
    #                 # self.state_content["post_atmosphere_wavefronts"][action_index].electric_field,
    #                 cmap='gray',
    #                 ax=ax)
    #             fig.colorbar(im, ax=ax)
                

    #             ###############################################################
    #             # Plot the atmopshere OPD, if an atmosphere is present.

    #             if self.state_content["atmos_layer_0_list"][action_index] is not None:
    #                 ax = plt.subplot2grid((num_rows, num_cols),
    #                                     (11, action_index),
    #                                     colspan=1,
    #                                     rowspan=1)
                    
    #                 ax.set_title('Atmosphere OPD [$\\mu$m] ($t$ = %.1f ms)' % action_time)
    #                 # plt.title('Atmos [$\\mu$m] ($t$ = %.3f ms)' % action_time)
    #                 layer = self.state_content["atmos_layer_0_list"][action_index]
    #                 phase_screen_phase = layer.phase_for(self.optical_system.wavelength) # in radian
    #                 phase_screen_opd = phase_screen_phase * (self.optical_system.wavelength / (2 * np.pi)) * 1e6 # in um
    #                 im = hcipy.imshow_field(
    #                     phase_screen_opd,
    #                     cmap='RdBu_r',
    #                     ax=ax)
    #                 fig.colorbar(im, ax=ax)
    #             ###############################################################
    #             # Plot Object Field pre-atmosphere.

    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (12, action_index),
    #                                   colspan=1,
    #                                   rowspan=1)
    #             ax.set_title('Object Field (propogated) [???] ($t$ = %.1f ms)' % action_time)
    #             # im = plt.imshow(
    #             #     self.state_content["object_fields"][action_index],
    #             #     # mask=self.optical_system.aperture,
    #             #     cmap='gray',
    #             #     # ax=ax
    #             # )
    #             im = hcipy.imshow_field(
    #                 self.state_content["pre_atmosphere_object_wavefronts"][action_index].intensity,
    #                 # self.state_content["pre_atmosphere_object_wavefronts"][action_index].electric_field,
    #                 # mask=self.optical_system.aperture,
    #                 cmap='gray',
    #                 ax=ax)
    #             fig.colorbar(im, ax=ax)
            
    #             ###############################################################
    #             # Plot Object Field illuminance (???).
    #             ax = plt.subplot2grid((num_rows, num_cols),
    #                                   (13, action_index),
    #                                   colspan=1,
    #                                   rowspan=1)
    #             ax.set_title('Object Field [???] ($t$ = %.1f ms)' % action_time)
    #             # im = plt.imshow(
    #             #     self.state_content["object_fields"][action_index],
    #             #     # mask=self.optical_system.aperture,
    #             #     cmap='gray',
    #             #     # ax=ax
    #             # )
    #             im = hcipy.imshow_field(
    #                 self.state_content["object_fields"][action_index],
    #                 # mask=self.optical_system.aperture,
    #                 cmap='gray',
    #                 ax=ax)
    #             fig.colorbar(im, ax=ax)
    #             ###############################################################

    #         plt.tight_layout()
    #         fig.set_dpi(self.render_dpi)
    #         fig.canvas.draw()

    #         rgb_image = np.array(fig.canvas.renderer.buffer_rgba())
    #         plt.close()
    #         return rgb_image

