"""
Distributed Aperture System for Interferometric Exploitation control system
simulation environment.

Author: Justin Fletcher
"""

import os


import gymnasium as gym
import glob
import time
import math
import copy
import uuid
import datetime
import numpy as np

from collections import deque

from scipy import signal
from anytree import Node, RenderTree


import scipy.ndimage as ndimage

from PIL import Image, ImageSequence

from gymnasium import spaces, logger
from gymnasium.utils import seeding

from matplotlib import pyplot as plt
from matplotlib import image
from matplotlib.figure import Figure

import pickle
from pathlib import Path


import astropy.units as u
import hcipy

def cosine_similarity(u, v):
    """

    :param u: Any np.array matching u in shape (and semantics probably)
    :param v: Any np.array matching v in shape (and semantics probably)
    :return: np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    """
    u = u.flatten()
    v = v.flatten()

    return (np.dot(v, u) / (np.linalg.norm(u) * np.linalg.norm(v)))

def gaussian_kernel(n, std, normalised=False):
    '''
    Generates a n x n matrix with a centered gaussian 
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    gaussian1D = signal.windows.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D

def offset_gaussian(n,
                    mu_x,
                    mu_y,
                    std,
                    kernel_extent,
                    normalised=False):

    

    scene_array = np.zeros((n, n))
    source_array = gaussian_kernel(kernel_extent,
                                    std,
                                    normalised=normalised)
    
    x_start = int(mu_x - kernel_extent // 2)
    x_end = int(mu_x + kernel_extent // 2)
    y_start = int(mu_y - kernel_extent // 2)
    y_end = int(mu_y + kernel_extent // 2)

    scene_array[x_start:x_end, y_start:y_end] = source_array

    return scene_array

def one_hot_array(n,
                  x,
                  y,
                  value=1.0):

    scene_array = np.zeros((n, n))
    scene_array[x, y] = value

    return scene_array


class ObjectPlane(object):

    def __init__(self,
                 object_type="binary",            
                 object_plane_extent_pixels=128,
                 object_plane_extent_meters=1.0,
                 object_plane_distance_meters=1000000, # m
                 randomize=False,
                 **kwargs):
        
        self.extent_pixels = object_plane_extent_pixels
        self.extent_meters = object_plane_extent_meters
        self.distance_meters = object_plane_distance_meters

        if object_type == "single":

            if randomize:

                kwargs['source_vmag'] = np.random.uniform(0.0, 25.0)
                kwargs['source_position'] = [
                    np.random.uniform(0.0, self.extent_pixels),
                    np.random.uniform(0.0, self.extent_pixels)  
                ]

            self.array = self.make_single_object(**kwargs)

        elif object_type == "binary":

            self.array = self.make_binary_object(**kwargs)

        elif object_type == "usaf1951":

            if object_plane_extent_pixels == 512:

                self.array = self.rgb2gray(image.imread('usaf1951_512.jpg'))

            elif object_plane_extent_pixels == 256:

                self.array = self.rgb2gray(image.imread('usaf1951_256.jpg'))

            elif object_plane_extent_pixels == 128:

                self.array = self.rgb2gray(image.imread('usaf1951_128.jpg'))

            else:

                raise NotImplementedError(
                    "ObjectPlane arg object_plane_extent was %s, but only \
                     512, 256, and 128 are \
                     implemented." % object_plane_extent_pixels)
            
            
            self.array = self.array / np.max(self.array)
            
            self.array = -(self.array - np.max(self.array))

            # self.array = np.flipud(self.array)

        elif object_type == "flat":

            self.array = np.ones((object_plane_extent_pixels, object_plane_extent_pixels))

        else:

            raise NotImplementedError(
                "ObjectPlane arg object_type was %s, but only \
                'single', 'binary', 'usaf1951' and 'flat' are implemented." % object_type)

        return
    
    def make_binary_object(self, **kwargs):


        # TODO: Obviously this needs to be externalized.
        # primary_source_vmag = kwargs['primary_source_vmag']
        # secondary_source_vmag = kwargs['secondary_source_vmag']
        # binary_separation = kwargs['binary_separation']
        # binary_position_angle = kwargs['binary_position_angle']
        
        std = 1.0
        kernel_extent = 8 * std
        # ifov (arcsec/pixel): 0.0165012 
        ifov = 0.0165012
        seperation_pixels = int(0.6 / ifov)
        mu_x = (self.extent_pixels // 2)
        mu_y = (self.extent_pixels // 2) - (seperation_pixels // 2)
        

        primary_source = offset_gaussian(self.extent_pixels,
                               mu_x,
                               mu_y,
                               std,
                               kernel_extent,
                               normalised=True)
        
        
        std = 1.0
        kernel_extent = 8 * std
        # ifov (arcsec/pixel): 0.0165012 
        ifov = 0.0165012
        seperation_pixels = int(0.6 / ifov)
        mu_x = (self.extent_pixels // 2)
        mu_y = (self.extent_pixels // 2) + (seperation_pixels // 2)

        secondary_source = offset_gaussian(self.extent_pixels,
                               mu_x,
                               mu_y,
                               std,
                               kernel_extent,
                               normalised=True)
        
        
        # print("seperation_pixels: %s" % seperation_pixels)
        
        return primary_source + secondary_source

    def make_single_object(self, **kwargs):

        # TODO: Externalize,
        # source_vmag = kwargs['source_vmag']
        # source_position = kwargs['source_position']

        # TODO: Correctly model a single objects intensity and translate to Gaussian std.
        # 0.02 arcsec fwhm
        # magnitude_response_function 16-bit value to: mag 9 star will give 5000 counts at 12 ms
        std = 1
        kernel_extent = 1

        # TODO: make mu_y and mu_x determined by source_position.
        # mu_x = self.extent_pixels // 4
        # mu_y = self.extent_pixels // 4
        x = self.extent_pixels // 2
        y = self.extent_pixels // 2

        return one_hot_array(self.extent_pixels,
                             x,
                             y,
                             value=1.0)

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


class OpticalSystem(object):

    def __init__(self, **kwargs):

        self.model_wind_diff_motion = kwargs["model_wind_diff_motion"]
        self.model_gravity_diff_motion = kwargs["model_gravity_diff_motion"]
        self.model_temp_diff_motion = kwargs["model_temp_diff_motion"]

        print(self.model_wind_diff_motion)
        self.report_time = kwargs["report_time"]
        self.microns_opd_per_actuator_bit = 0.00015
        # self.num_apertures = 15
        # self.num_apertures = 2

        self.num_tensioners = kwargs["num_tensioners"]

        self.model_ao = kwargs['model_ao']

        aperture_type = kwargs['aperture_type']
        # Build the selected aperture; elf is the default.
        # TODO: Someday we should generalize this to accept any HCIPy aperture.
        print("Building aperture.")
        if aperture_type == "elf":

            # The sELF FOV is narrow (about 10 arcsec) so that its volume is small 
            # and it can use optically fast primary and small secondary mirrors.
            # The telescope has an effective focal length of 32.5m at focal 
            # ratio F/8.25. The outer circumscribing radius of the M1 subapertures 
            # is 3.46m with an effective fill-factor of 0.3.

            self.num_apertures = 15


            # # TODO: Externalize.
            # kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-3
            kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-4


            # Parameters for the pupil function
            # focal_length = 32.5 # m
            focal_length = 32.5 # m
            pupil_diameter = 3.6 # m
            segment_diameter = 0.5
            elf_segment_centroid_diameter = 2.7 # m

            # Initialize an empty structual interaction matrix.
            self.optomech_interaction_matrix = None
            interaction_size = 1
            self._optomech_encoder = np.random.rand(self.num_tensioners, interaction_size)
            self._optomech_decoder = np.random.rand(interaction_size, self.num_apertures * 3)


            # Instantiate a segmented aperture.
            aperture, segments = self.make_elf_aperture(
                pupil_diameter=elf_segment_centroid_diameter,
                num_apertures=self.num_apertures,
                segment_diameter=segment_diameter,
                return_segments=True,
            )

        elif aperture_type == "circular":


            # Parameters for the pupil function
            focal_length = 200.0 # m
            pupil_diameter = 3.6 # m
            elf_segment_centroid_diameter = 2.5 # m

            # # TODO: Externalize.
            # kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-3
            kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-4


            # Instantiate a circular aperture.
            aper_coords = hcipy.SeparatedCoords(
                (np.array([0.0]), np.array([0.0]))
            )
            segment_centers = hcipy.PolarGrid(aper_coords)
            aperture = hcipy.make_circular_aperture(
                elf_segment_centroid_diameter
            )
            segments = hcipy.make_segmented_aperture(
                aperture,
                segment_centers,
                return_segments=True
            )

        elif aperture_type == "nanoelf":


            self.num_apertures = 2

            # Parameters for the pupil function
            focal_length = 1.018 # m
            pupil_diameter = 0.1408 # m

            # # TODO: Externalize.
            # kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-3
            kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-5

    
            # Instantiate a segmented aperture.
            aperture, segments = self.make_nanoelf_aperture(
                pupil_diameter=pupil_diameter / 2.0,
                num_apertures=self.num_apertures,
                segment_diameter=0.0254 * 2,
                return_segments=True,
            )

        elif aperture_type == "nanoelfplus":


            # Parameters for the pupil function
            focal_length = 1.018 # m
            pupil_diameter = 0.1408 # m
            self.num_apertures = 3

            # # TODO: Externalize.
            # kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-3
            kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-5

    
            # Instantiate a segmented aperture.
            aperture, segments = self.make_nanoelf_aperture(
                pupil_diameter=pupil_diameter / 2.0,
                num_apertures=self.num_apertures,
                segment_diameter=0.0254 * 2,
                return_segments=True,
            )

        else:

            # Note: if you add aperture types, update this exception.
            raise NotImplementedError(
                "aperture_type was %s, but only 'elf' and 'circular' are \
                implemented." % aperture_type)


        
        # Parameters for the optical simulation.
        num_pupil_grid_simulation_pixels = kwargs['focal_plane_image_size_pixels']
        # self.wavelength = 763e-9
        self.wavelength = 1000e-9
        oversampling_factor = 8

        # Parameters for the atmosphere. Caution: better seeing = longer runs.
        # TODO: Externalize.
        seeing = 0.5 # arcsec @ 500nm (convention)
        # seeing = 0.0001 # arcsec @ 500nm (convention)
        outer_scale = 40 # meter
        # Greenwood time constant seconds
        tau0 = 10.0

        self.simulate_differential_motion = kwargs['simulate_differential_motion']
        self.init_differential_motion = kwargs['init_differential_motion']

        # Parameters for the structual wind response model.
        # TODO: Externalize.
        initial_ground_wind_speed_mps = 3.0
        # initial_ground_wind_speed_mps = 16.7
        # The std of ground wind speed in m/s over one ms, joffre1988standard.
        self.ground_wind_speed_ms_sampled_std_mps = 0.08 * initial_ground_wind_speed_mps
        self.ground_wind_speed_mps = initial_ground_wind_speed_mps
        
        # Parameters for the structual temperature response model.
        # TODO: Externalize.
        initial_ground_temp_degcel = 20.0
        # The std of ground tempertature in celcius over one millisecond.
        self.ground_temp_ms_sampled_std_mps = 0.0
        self.ground_temp_degcel = initial_ground_temp_degcel

        # TODO: Resolve and remove.
        if self.ground_temp_ms_sampled_std_mps != 0.0:
            raise NotImplementedError(
                "Non-zero ground_temp_ms_sampled_std_mps is not supported.")

        # Parameters for the structual gravity response model.
        # TODO: Externalize.
        initial_gravity_normal_deg = 45.0
        # The std of 1d gravity normal in deg over one millisecond.
        self.gravity_normal_ms_sampled_std_mps = 0.0
        self.gravity_normal_deg = initial_gravity_normal_deg

        # TODO: Resolve and remove.
        if self.gravity_normal_ms_sampled_std_mps != 0.0:
            raise NotImplementedError(
                "Non-zero gravity_normal_ms_sampled_std_mps is not supported.")

        fried_parameter = hcipy.seeing_to_fried_parameter(seeing)
        print("fried_parameter: %s" % fried_parameter)
        Cn_squared = hcipy.Cn_squared_from_fried_parameter(
            fried_parameter,
            wavelength=self.wavelength
        )
        velocity = 0.314 * fried_parameter / tau0

        # Extract the provided focal plane pararmeters.
        # TODO: Refactor to add a separate focal grid for the SHWFS.
        num_focal_grid_pixels = kwargs['focal_plane_image_size_pixels']
        focal_plane_extent_metres = kwargs['focal_plane_image_size_meters']

        airy_extent_radians = 1.22 * self.wavelength / pupil_diameter
        airy_extent_meters = airy_extent_radians * focal_length
        focal_plane_pixel_extent_meters = focal_plane_extent_metres / num_focal_grid_pixels
        sampling = airy_extent_meters / focal_plane_pixel_extent_meters
        sampling = airy_extent_meters / focal_plane_pixel_extent_meters

        focal_plane_resolution_element = self.wavelength * focal_length / pupil_diameter
        focal_plane_pixels_per_meter = num_focal_grid_pixels / focal_plane_extent_metres
        focal_plane_pixel_extent_meters = focal_plane_extent_metres /  num_focal_grid_pixels
        self.ifov = (206265 / focal_length) * focal_plane_pixel_extent_meters
        fov = self.ifov * num_focal_grid_pixels

        # self.ifov = 10.0 / num_focal_grid_pixels

        # The sELF FOV is narrow (about 10 arcsec) so that its volume is small 
        # and it can use optically fast primary and small secondary mirrors.
        # The telescope has an effective focal length of 32.5m at focal 
        # ratio F/8.25. The outer circumscribing radius of the M1 subapertures 
        # is 3.46m with an effective fill-factor of 0.3.


        # sampling: The number of pixels per resolution element (= lambda f / D).
        # sampling = focal_plane_resolution_element / focal_plane_pixel_extent_meters
        # num_airy: The spatial extent of the grid in radius in resolution elements (= lambda f / D).
        num_airy = num_focal_grid_pixels / (2 * sampling)


        # Log computed values to aid in debugging.
        print("num_focal_grid_pixels: %s" % num_focal_grid_pixels)
        print("focal_plane_extent_metres: %s" % focal_plane_extent_metres)
        print("focal_plane_pixel_extent_meters: %s" % focal_plane_pixel_extent_meters)
        print("focal_plane_resolution_element: %s" % focal_plane_resolution_element)
        print("focal_plane_pixels_per_meter: %s" % focal_plane_pixels_per_meter)
        print("num_airy (The spatial extent of the grid in radius in resolution elements): %s" % num_airy)
        print("sampling (number of pixels per resolution element): %s" % sampling)
        print("ifov (arcsec/pixel): %s" % self.ifov)
        print("fov (arcsec): %s" % fov)

        # Build the object plane for this system.
        print("Building object plane.")
        object_plane_extent_meters = 1.0
        object_plane_distance_meters = 1.0
        object_type=kwargs['object_type']
        self.object_plane = ObjectPlane(
            object_type=object_type,
            object_plane_extent_pixels=num_pupil_grid_simulation_pixels,
            object_plane_extent_meters=object_plane_extent_meters,
            object_plane_distance_meters=object_plane_distance_meters
        )

        # Make the simulation grid for the pupil plane.
        print("Building pupil grid.")
        self.pupil_grid = hcipy.make_pupil_grid(
            dims=num_pupil_grid_simulation_pixels,
            diameter=pupil_diameter
        )

        # Initialize a list to store the atmosphere layers.
        self.atmosphere_layers = list()

        # Add the atmosphere layers.
        print("Building atmosphere layers.")
        for layer_num in range(kwargs['num_atmosphere_layers']):

            layer = hcipy.InfiniteAtmosphericLayer(self.pupil_grid,
                                                   Cn_squared,
                                                   outer_scale,
                                                   velocity)
            
            self.atmosphere_layers.append(layer)
        
        # Make the simulation grid for the focal plane.
        # focal_grid = hcipy.make_focal_grid(
        #     sampling,
        #     num_airy,
        #     spatial_resolution=self.wavelength * focal_length / pupil_diameter,
        # )

        focal_grid = hcipy.make_pupil_grid(
            dims=num_pupil_grid_simulation_pixels,
            diameter=focal_plane_extent_metres
        )
        focal_grid = focal_grid.shifted(focal_grid.delta / 2)

        # Instantiate a Fraunhofer propagator from the pupil to focal plane.
        print("Building pupil to focal propagator.")
        self.pupil_to_focal_propagator = hcipy.FraunhoferPropagator(
            self.pupil_grid,
            focal_grid, 
            focal_length
        )

        

        # Evaluate the aperture and segments, initializing them.
        aperture = hcipy.evaluate_supersampled(aperture,
                                                self.pupil_grid,
                                                oversampling_factor)
        segments = hcipy.evaluate_supersampled(segments,
                                                self.pupil_grid,
                                                oversampling_factor)
        
        # Unite the segments into a single mirror for low order modeling.
        self.segmented_mirror = hcipy.SegmentedDeformableMirror(segments)
        self.aperture = aperture

        wavefront = hcipy.Wavefront(self.aperture, self.wavelength)
        perfect_image = self.pupil_to_focal_propagator(self.segmented_mirror(wavefront))
        self.perfect_image = perfect_image.intensity

        # TODO: Externalize and modularize for other WFS types, including none.
        # Instantiate a Shack-Hartmann wavefront sensor.
        print("Building Shack-Hartmann wavefront sensor.")
        f_number = 50
        num_lenslets = 40 # 40 lenslets along one diameter
        sh_diameter = 5e-3 # m

        magnification = sh_diameter / pupil_diameter
        self.magnifier = hcipy.Magnifier(magnification)
        self.shwfs = hcipy.SquareShackHartmannWavefrontSensorOptics(
            self.pupil_grid.scaled(magnification),
            f_number,
            num_lenslets,
            sh_diameter)
        self.shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(
            self.shwfs.mla_grid,
            self.shwfs.micro_lens_array.mla_index
        )

        # TODO: Refactor to add a separate focal grid for the SHWFS.
        self.shwfs_camera = hcipy.NoiselessDetector(focal_grid)

        # Instantiate a deformable mirror.
        if self.model_ao:
            print("Building deformable mirror.")
            dm_model_type = "gaussian_influence"

            if dm_model_type == "disk_harmonic_basis":
                num_modes = 500
                dm_modes = hcipy.make_disk_harmonic_basis(
                    self.pupil_grid,
                    num_modes,
                    pupil_diameter,
                    'neumann'
                )
                dm_modes = hcipy.ModeBasis(
                    [mode / np.ptp(mode) for mode in dm_modes],
                    self.pupil_grid
                )
                self.dm = hcipy.DeformableMirror(dm_modes)

            elif dm_model_type == "gaussian_influence":
                self.dm_influence_functions = hcipy.make_gaussian_influence_functions(
                    self.pupil_grid,
                    num_actuators_across_pupil=35,
                    actuator_spacing=pupil_diameter / 35
                )

                self.dm = hcipy.DeformableMirror(self.dm_influence_functions)

        # Initialize natural structural differential motion.
        if self.init_differential_motion:

            print("Initializing natural differential motion.")
            self._init_natural_diff_motion()
        
        # Store the baseline segment displacements.
        self._store_baseline_segment_displacements()
        

        # Finally, make a camera.
        # Note: The camera is noiseless here; we add noise in the Env step().
        print("Building camera.")
        self.camera = hcipy.NoiselessDetector(focal_grid)

        print("Optical system initialized.")

    

    def make_object_field(self, array, center=None):
        '''Makes a Field generator for an object plane.

        Parameters
        ----------
        extent : array
            A numpy array representing the field intensity at each point.
        center : array_like
            The center of the field

        Returns
        -------
        Field generator
            This function can be evaluated on a grid to get a Field.
        '''

        def func(grid):
            if grid.is_separated:
                x, y = grid.separated_coords
                x = x[np.newaxis, :]
                y = y[:, np.newaxis]
            else:
                x, y = grid.coords

            f = array.ravel()

            return hcipy.Field(f.astype('float'), grid)

        return func


    def evolve_atmosphere_to(self, episode_time_ms):

        # Compute the time in seconds.
        episode_time_seconds = episode_time_ms / 1000.0

        for layer in self.atmosphere_layers:

            # TODO: Major Feature. This is the Amdahl op. Replace it by caching a set
            #       of atmosphere layers at a variety of strengths, queuing them from
            #       disk in parallel, and reading them from the queue in time order.
            layer.evolve_until(episode_time_seconds)

        return


    def command_dm(self, dm_command):

        # meters_opd_per_actuator_bit = self.microns_opd_per_actuator_bit * 1e-6
        # command_vector = command_grid.flatten().astype(np.float64)

        # Compute the stroke limit of the DM.
        dm_stroke_meters = self.stroke_count_limit * self.microns_opd_per_actuator_bit * 1e-6

        # Unpack the DM command into a command vector with range [-1, 1].
        command_vector = np.array([x[0] for x in dm_command])

        # Scale the command vector to the phyiscal stroke limit of the DM.
        dm_command_meters = (dm_stroke_meters / 2.0) * command_vector

        # Set the dm actuators to the command vector, in meters.
        self.dm.actuators = dm_command_meters

        return
    

    def simulate(self):

        """
        
        Simulate the optical system.

        This function simulates the optical system by propagating a wavefront
        through the system, including the atmosphere, segmented mirror, DM, and
        focal plane. The function also simulates natural differential motion
        of the segmented mirror, if enabled.

        """

        # Make a pupil plane wavefront from aperture
        object_wavefront_start_time = time.time()


        self.object_wavefront = hcipy.Wavefront(self.aperture,
                                                self.wavelength)
        self.pre_atmosphere_object_wavefront = self.object_wavefront

        
        if self.report_time:
            print("--- Object Wavefront time: %0.6f" % (time.time() - object_wavefront_start_time))

        atmosphere_forward_start_time = time.time()

        # Propagate the object plane wavefront through the atmosphere layers.
        wf = self.pre_atmosphere_object_wavefront
        for atmosphere_layer in self.atmosphere_layers:
            wf = atmosphere_layer.forward(wf)
        self.post_atmosphere_wavefront = wf

        if self.report_time:
            print("--- Atmosphere Forward time: %0.6f" % (time.time() - atmosphere_forward_start_time))

        if self.simulate_differential_motion:

            natural_diff_motion_start_time = time.time()

            # Simulate natural differential motion of the segmented mirror.
            self._simulate_natural_diff_motion()

            if self.report_time:
                print("--- Natural Diff Motion time: %0.6f" % (time.time() - natural_diff_motion_start_time))


        segmented_mirror_forward_start_time = time.time()
        # Apply the segmented mirror pupil to the post-atmosphere wavefront.
        self.pupil_wavefront = self.segmented_mirror(
            self.post_atmosphere_wavefront
        )

        if self.report_time:
            print("--- Segments Forward time: %0.6f" % (time.time() - segmented_mirror_forward_start_time))

        if self.model_ao:
            dm_forawrd_start_time = time.time()
            # Propagate the wavefront from the segmented mirror through the DM.
            # Note: counter-intuitively, the DM must be re-applied after changes.
            self.post_dm_wavefront = self.dm.forward(self.pupil_wavefront)

            if self.report_time:

                print("--- DM Forward time: %0.6f" % (time.time() - dm_forawrd_start_time))
        else:
            self.post_dm_wavefront = self.pupil_wavefront

        pupil_focal_prop_start_time = time.time()

        # Propagate from the DM (M2) to the focal (image) plane.
        # TODO: rename pupil_to_focal_propagator to dm_to_focal_propagator
        # NOTE: This is the ahmdal op: the longest-running command.
        self.focal_plane_wavefront = self.pupil_to_focal_propagator(
            self.post_dm_wavefront
        )

        if self.report_time:
            print("--- Pupil-Focal Propagation time: %0.6f" % (time.time() - pupil_focal_prop_start_time))
    

    def get_shwfs_frame(self, integration_seconds=1.0):

        self.shwfs_camera.integrate(self.shwfs(self.magnifier(self.post_dm_wavefront)), integration_seconds)
        shwfs_readout_image = self.shwfs_camera.read_out()

        return shwfs_readout_image
    

    def calibrate_dm_interaction_matrix(self, env_uuid):

        probe_amp = 0.01 * self.wavelength
        response_matrix = list()

        print("Calibrating DM.")

        # If a DM interaction matrix pickle has already been computed, load it.
        # Create save directory if it doesn't already exist.
        dm_cache_path = os.path.join(
                "./tmp/cache/",
                str(env_uuid),
            )
        
        if os.path.exists(dm_cache_path):
                print("Found Cached Interaction Matrix.")
                
                with open(os.path.join(dm_cache_path,
                          'dm_interaction_matrix.pkl'), 'rb') as f:
                    self.interaction_matrix = pickle.load(f)
                    return


        # First, take a reference image, which is just the aperture.
        wf = hcipy.Wavefront(self.aperture, self.wavelength)
        wf.total_power = 1
        self.shwfs_camera.integrate(self.shwfs(self.magnifier(wf)), 1)
        reference_image = self.shwfs_camera.read_out()

        # Adjust our SHWFS estimator to account for the reference image.
        fluxes = ndimage.measurements.sum(reference_image,
                                          self.shwfse.mla_index,
                                          self.shwfse.estimation_subapertures)
        flux_limit = fluxes.max() * 0.5
        estimation_subapertures = self.shwfs.mla_grid.zeros(dtype='bool')
        estimation_subapertures[self.shwfse.estimation_subapertures[fluxes > flux_limit]] = True
        self.shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid,
                                                                  self.shwfs.micro_lens_array.mla_index,
                                                                  estimation_subapertures)

        # Compute reference image slopes.
        self.reference_slopes = self.shwfse.estimate([reference_image])

        for i in range(len(self.dm.actuators)):

            print("Calibrating DM actuator %s." % i)
            
            slope = 0

            # Probe the phase response
            amps = [-probe_amp, probe_amp]
            for amp in amps:
                self.dm.flatten()
                self.dm.actuators[i] = amp

                dm_wf = self.dm.forward(wf)
                dm_wf.total_power = 1
                wfs_wf = self.shwfs(self.magnifier(dm_wf))

                self.shwfs_camera.integrate(wfs_wf, 1)
                image = self.shwfs_camera.read_out()

                slopes = self.shwfse.estimate([image])

                slope += amp * slopes / np.var(amps)

            response_matrix.append(slope.ravel())

        self.interaction_matrix = hcipy.ModeBasis(response_matrix)

        Path(dm_cache_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dm_cache_path, "dm_interaction_matrix.pkl"), 'wb') as f:
            pickle.dump(self.interaction_matrix, f)


    def get_science_frame(self, integration_seconds=1.0):

        integration_start_time = time.time()
        # Integrate the wavefront, read out, and return. 
        self.camera.integrate(self.focal_plane_wavefront, integration_seconds)

        if self.report_time:
            print("-- Camera Integration time: %0.6f" % (time.time() - integration_start_time))

        use_geometric_optics = True
        if use_geometric_optics:

            
            read_out_start_time = time.time()
            # This is the effective PSF of the system, as the camera is noiseless.
            effective_psf = self.camera.read_out()

            effective_psf = effective_psf.reshape(
                (
                    int(np.sqrt(effective_psf.size)),
                    int(np.sqrt(effective_psf.size))
                )
            )

            self.instantaneous_psf = effective_psf

            if self.report_time:
                print("-- Readout time: %0.6f" % (time.time() - read_out_start_time))

            fft_start_time = time.time()
            # Compute the effective OTF of the system.
            effective_otf = np.fft.fft2(effective_psf)
            
            # Compute the spectrum of the object.
            object_spectrum = np.fft.fft2(self.object_plane.array)

            # Hadamard product system OTF to object spectrum, making the image spectrum.
            image_spectrum = object_spectrum * effective_otf

            # Compute the image.
            self.readout_image = np.abs(np.fft.fftshift(np.fft.ifft2(image_spectrum)))
            if self.report_time:
                print("-- FFT time: %0.6f" % (time.time() - fft_start_time))

        else:
            self.readout_image = self.camera.read_out()

        return self.readout_image


    def make_elf_aperture(self,
                          pupil_diameter=2.5,
                          num_apertures=15,
                          segment_diameter=0.5,
                          return_segments=True,
                          **kwargs):

        """
        Create a ExoLife finder segmented aperture.


        Parameters
        ----------
        pupil_diameter : float
            The edge-to-edge diameter of the pupil in meters. TODO: Update
        num_segments : int
            The number of segments to add to the aperture 
        segment_diameter : float
            The diameter of each circular segment in meters.

        return_segments : boolean
            Whether to return a ModeBasis of all segments as well.

        Returns
        -------
        Field generator
            The segmented aperture.
        list of Field generators
            The segments. Only returned if return_segments is True.
        """

        pupil_radius = pupil_diameter / 2
        segments = list()

        # Linear space of angular coordinates for mirror centers
        segment_angles = np.linspace(0, 2 * np.pi, num_apertures + 1)[:-1]

        # Use HCIPy coordinate generation to generate mirror centers
        aper_coords = hcipy.SeparatedCoords(
                (np.array([pupil_radius]), segment_angles)
        )

        # Create an HCIPy "CartesianGrid" by creating PolarGrid and converting
        segment_centers = hcipy.PolarGrid(aper_coords).as_('cartesian')

        # Build a circular aperture.
        aperture = hcipy.make_circular_aperture(
                segment_diameter,
        )       

        # Place copies of the circular aperture at the segment centers.
        aperture, segments = hcipy.make_segmented_aperture(
            aperture,
            segment_centers,
            return_segments=return_segments
        )

        if return_segments:

            return aperture, segments
        
        else:

            return aperture


    def make_nanoelf_aperture(self,
                              pupil_diameter=2.5,
                              num_apertures=15,
                              segment_diameter=0.5,
                              return_segments=True,
                              **kwargs):

        """
        Create a ExoLife finder segmented aperture.


        Parameters
        ----------
        pupil_diameter : float
            The edge-to-edge diameter of the pupil in meters. TODO: Update
        num_segments : int
            The number of segments to add to the aperture 
        segment_diameter : float
            The diameter of each circular segment in meters.

        return_segments : boolean
            Whether to return a ModeBasis of all segments as well.

        Returns
        -------
        Field generator
            The segmented aperture.
        list of Field generators
            The segments. Only returned if return_segments is True.
        """

        pupil_radius = pupil_diameter / 2
        segments = list()

        # Linear space of angular coordinates for mirror centers
        segment_angles = np.linspace(0, 2 * np.pi, num_apertures + 1)[:-1]

        # Use HCIPy coordinate generation to generate mirror centers
        aper_coords = hcipy.SeparatedCoords(
                (np.array([pupil_radius]), segment_angles)
        )

        # Create an HCIPy "CartesianGrid" by creating PolarGrid and converting
        segment_centers = hcipy.PolarGrid(aper_coords).as_('cartesian')

        # Build a circular aperture.
        aperture = hcipy.make_circular_aperture(
                segment_diameter,
        )       

        # Place copies of the circular aperture at the segment centers.
        aperture, segments = hcipy.make_segmented_aperture(
            aperture,
            segment_centers,
            return_segments=return_segments
        )

        if return_segments:

            return aperture, segments
        
        else:

            return aperture

    def _optomechanical_interaction(self, tension_forces):

        """
        This function maps the current state of the tensioners to an a optical
        figure modification expressed as a global offset.
        """

        # Convert the tuple of tension force arrays to a numpy array.
        tension_forces = np.transpose(np.array(tension_forces))

        # A simple, random MLP approximation of the optomechanical interaction.
        # TODO: Add a weight import functionality in the constructor.
        # TODO: Add a direct tensegrity simulator call here.
        # optomech_embedding = np.tanh(tension_forces.dot(self._optomech_encoder))
        # optomech_ptt_displacements = np.tanh(optomech_embedding.dot(self._optomech_decoder))
        optomech_embedding = tension_forces.dot(self._optomech_encoder)
        optomech_ptt_displacements = optomech_embedding.dot(self._optomech_decoder)

        # Reshape the outputs to model three displacements per aperture.
        optomech_ptt_displacements = optomech_ptt_displacements.reshape((self.num_apertures, 3))

        # TODO: this turns off the optomechanical interaction.
        optomech_ptt_displacements = np.zeros((self.num_apertures, 3))


        # Convert the displacements to physical units
        optomech_ptt_displacements[:, 0] *= 1e-6
        optomech_ptt_displacements[:, 1] *= np.pi / (180 * 3600)
        optomech_ptt_displacements[:, 2] *= np.pi / (180 * 3600)

        # Limit the displacements to the phsyical range of the optomechanical system.
        # optomech_ptt_displacements[:, 0] = np.clip(optomech_ptt_displacements[:, 0], -1.0, 1.0)


        # Apply the displacements.
        self._apply_ptt_displacements(ptt_displacements=optomech_ptt_displacements)

        return
    

    def _simulate_natural_diff_motion(self):
        
        """
        Simulate natural differential motion of the segmented mirror.

        This function simulates the natural differential motion of the
        segmented mirror due to various environmental factors. This function
        assumes an observation scenario in which the aperture is already
        pointed at a stationary (though not necessesarily static) scene, and 
        all structural deflection is caused by the telescope settling into its
        pointing configuration, changes in temperature, and variation in wind 
        angle and speed. 

        Deflections are modeled as a limited random walk through piston, tip, 
        and tilt space.

        TODO: Need opinion on time-scale dynamics.
        
        
        Parameters
        ----------
        natural_diff_motion_piston_std : float
            The standard deviation of the piston displacements in microns.
        natural_diff_motion_tip_std : float
            The standard deviation of the tip displacements in microns.
        natural_diff_motion_tilt_std : float
            The standard deviation of the tilt displacements in microns.

        
        """
        # Generate a sample of ptt displacements.
        # ptt_displacements = np.random.randn(self.num_apertures, 3)
        # Microns to meters 1.0 -> 1e-6

        if self.model_wind_diff_motion:
            # TODO: Modularize as a TelescopeEnvironment class.
            # Update the windspeed.
            # TODO: Add a time-scale to the wind speed evolution.
            self.ground_wind_speed_mps += np.random.randn() * self.ground_wind_speed_ms_sampled_std_mps
            # Apply some limits to wind speed.
            if self.ground_wind_speed_mps < 0.0:
                self.ground_wind_speed_mps = 0.0
            if self.ground_wind_speed_mps > 20.0:
                self.ground_wind_speed_mps = 20.0

            # Compute and apply the wind displacments.
            # TODO: Compute these values. Need help from Tim and Ye.
            # TODO: this is completely made up. Replace with real estimator.
            wind_diff_motion_piston_micron_std = (self.ground_wind_speed_mps / 8) 
            wind_diff_motion_tip_arcsec_std = (self.ground_wind_speed_mps / 32) 
            wind_diff_motion_tilt_arcsec_std = (self.ground_wind_speed_mps / 32) 
            wind_ptt_displacements = np.random.randn(self.num_apertures, 3)
            # Sample displacement in meters.
            wind_ptt_displacements[:, 0] *= wind_diff_motion_piston_micron_std * 1e-6
            # # Sample displacement in radians.
            wind_ptt_displacements[:, 1] *= wind_diff_motion_tip_arcsec_std * np.pi / (180 * 3600)
            wind_ptt_displacements[:, 2] *= wind_diff_motion_tilt_arcsec_std * np.pi / (180 * 3600)
            wind_incremental_factor = 0.01
            self._apply_ptt_displacements(wind_ptt_displacements,
                                          incremental=True,
                                          incremental_factor=wind_incremental_factor)
        

        if self.model_temp_diff_motion:

            # # Compute and apply the temperature displacments.
            # self.ground_temp_ms_sampled_std_mps
            # self.ground_temp_degcel
            # # TODO: Compute these values. Need help from Tim and Ye.
            temp_ptt_displacements = np.random.randn(self.num_apertures, 3)
            # # Sample displacement in meters.
            # # TODO: This will always be 0.0 for now.
            # temp_ptt_displacements[:, 0] *= self.ground_temp_ms_sampled_std_mps * 1e-6
            # # Sample displacement in radians.
            # temp_ptt_displacements[:, 1] *= self.ground_temp_ms_sampled_std_mps * np.pi / (180 * 3600)
            # temp_ptt_displacements[:, 2] *= self.ground_temp_ms_sampled_std_mps * np.pi / (180 * 3600)
            # self._apply_ptt_displacements(temp_ptt_displacements)

        if self.model_gravity_diff_motion:

            # # Compute and apply the gravity displacments.
            # self.gravity_normal_ms_sampled_std_mps
            # self.gravity_normal_deg
            # # TODO: Compute these values. Need help from Tim and Ye.
            gravity_ptt_displacements = np.random.randn(self.num_apertures, 3)
            # # TODO: This will always be 0.0 for now.
            # # Sample displacement in meters.
            # gravity_ptt_displacements[:, 0] *= self.gravity_normal_ms_sampled_std_mps * 1e-6
            # # Sample displacement in radians.
            # gravity_ptt_displacements[:, 1] *= self.gravity_normal_ms_sampled_std_mps * np.pi / (180 * 3600)
            # gravity_ptt_displacements[:, 2] *= self.gravity_normal_ms_sampled_std_mps * np.pi / (180 * 3600)
            # self._apply_ptt_displacements(gravity_ptt_displacements)
        

        return
    

    def _init_natural_diff_motion(self):

        """
        Initialize the natural differential motion of the segmented mirror.

        This method uses the provided or default structural and environmental
        parameters to initialize the natural differential motion of the 
        primary segments. It maps the provided parameters to the piston, tip,
        and tilt displacements of the primary segments, then applies these 
        displacements.
        """

        if not(self.model_wind_diff_motion) and \
            not(self.model_temp_diff_motion) and \
            not(self.model_gravity_diff_motion):

            raise ValueError("You initialized differential motion, but no " + \
                             "types of differential motion are selected " + \
                             "for modeling. This may produce undesired " + \
                             "results.")

        if self.model_wind_diff_motion:

            # Compute and apply the wind displacments.
            self.ground_wind_speed_ms_sampled_std_mps
            self.ground_wind_speed_mps
            # TODO: Compute these values. Need help from Tim and Ye.
            # TDOD: Validate "3 sigma" assuption here
            wind_diff_motion_piston_micron_std = 1.0 / 3
            wind_diff_motion_tip_arcsec_std = 0.25 / 3 
            wind_diff_motion_tilt_arcsec_std = 0.25 / 3
            wind_ptt_displacements = np.random.randn(self.num_apertures, 3)
            # Sample displacement in meters.
            wind_ptt_displacements[:, 0] *= wind_diff_motion_piston_micron_std * 1e-6
            # Sample displacement in radians.
            wind_ptt_displacements[:, 1] *= wind_diff_motion_tip_arcsec_std * np.pi / (180 * 3600)
            wind_ptt_displacements[:, 2] *= wind_diff_motion_tilt_arcsec_std  * np.pi / (180 * 3600)
            self._apply_ptt_displacements(wind_ptt_displacements)


        if self.model_temp_diff_motion:
            # Compute and apply the temperature displacments.
            self.ground_temp_ms_sampled_std_mps
            self.ground_temp_degcel
            # TODO: Compute these values. Need help from Tim and Ye.
            temp_diff_motion_piston_micron_std = 0.0
            temp_diff_motion_tip_arcsec_std = 0.0
            temp_diff_motion_tilt_arcsec_std = 0.0
            temp_ptt_displacements = np.random.randn(self.num_apertures, 3)
            # Sample displacement in meters.
            temp_ptt_displacements[:, 0] *= temp_diff_motion_piston_micron_std * 1e-6
            # Sample displacement in radians.
            temp_ptt_displacements[:, 1] *= temp_diff_motion_tip_arcsec_std  * np.pi / (180 * 3600)
            temp_ptt_displacements[:, 2] *= temp_diff_motion_tilt_arcsec_std  * np.pi / (180 * 3600)
            self._apply_ptt_displacements(temp_ptt_displacements)


        if self.model_gravity_diff_motion:
            # Compute and apply the gravity displacments.
            self.gravity_normal_ms_sampled_std_mps
            self.gravity_normal_deg
            # TODO: Compute these values. Need help from Tim and Ye.
            gravity_diff_motion_piston_micron_std = 300.0
            gravity_diff_motion_tip_arcsec_std = 15.0
            gravity_diff_motion_tilt_arcsec_std = 15.0
            gravity_ptt_displacements = np.random.randn(self.num_apertures, 3)
            # Sample displacement in meters.
            gravity_ptt_displacements[:, 0] *= gravity_diff_motion_piston_micron_std * 1e-6
            # # Sample displacement in radians.
            gravity_ptt_displacements[:, 1] *= gravity_diff_motion_tip_arcsec_std * np.pi / (180 * 3600) 
            gravity_ptt_displacements[:, 2] *= gravity_diff_motion_tilt_arcsec_std * np.pi / (180 * 3600)
            self._apply_ptt_displacements(gravity_ptt_displacements)
    
  
    def _apply_ptt_displacements(self,
                                 ptt_displacements,
                                 incremental=False,
                                 incremental_factor=1.0):

        """
        Apply the provided incremental PTT displacements to the segmented
        mirror.


        Parameters
        ----------
        ptt_displacements : np.ndarray
            An array of shape (num_apertures, 3) containing the commanded
            PTT displacements to be applied to each segment. Each displacement
            is to be provided in microns, while both tip and tilt are to be 
            given in arcseconds.
        """

        # Iterate over each segment, applying the provided displacements.
        for segment_id in range(self.num_apertures):
            # print("Segment ID: %s" % segment_id)

            if incremental:

                # First, get the ptt displacements in meters of of this segement...
                (segment_piston,
                 segment_tip,
                 segment_tilt) = self.segmented_mirror.get_segment_actuators(segment_id)

                segment_piston_displacement = segment_piston + (incremental_factor * ptt_displacements[segment_id, 0])
                segment_tip_displacement = segment_tip + (incremental_factor * ptt_displacements[segment_id, 1])
                segment_tilt_displacement = segment_tilt + (incremental_factor * ptt_displacements[segment_id, 2])

            else:

                # TODO: the following three lines need be generalized to nested lists.
                segment_piston_displacement = ptt_displacements[segment_id, 0]
                segment_tip_displacement = ptt_displacements[segment_id, 1]
                segment_tilt_displacement = ptt_displacements[segment_id, 2]


            # Set the actuators in meter and radians.
            self.segmented_mirror.set_segment_actuators(
                segment_id,
                segment_piston_displacement,
                segment_tip_displacement,
                segment_tilt_displacement
            )


    def command_tensioners(self, tensioner_commands):

        direct_tension_command = True

        if direct_tension_command:

            tension_forces = tensioner_commands

        else:

            tension_forces += tensioner_commands

        self._optomechanical_interaction(tension_forces)

        return


    def _store_baseline_segment_displacements(self):

        self.segment_baseline_dict = dict()

        for segment_id in range(self.num_apertures):

            (segment_piston,
             segment_tip,
             segment_tilt) = self.segmented_mirror.get_segment_actuators(segment_id)

            self.segment_baseline_dict[segment_id] = {
                "piston": segment_piston,
                "tip": segment_tip,
                "tilt": segment_tilt
            }



    def command_secondaries(self, secondaries_commands):

        """
        Command the secondaries to a new state using the provided commands.

        Parameters
        ----------
        ptt_displacements : np.ndarray
            An array of shape (num_apertures, 3) containing the commanded
            PTT displacements to be applied to each segment. These are scaled
            from 0.0 to 1.0, and must be converted to physical units here.
        
        """
        
        # secondaries_ptt_displacements = secondaries_commands


        segments_ptt_commands = secondaries_commands
    
        # Cannonical
        max_piston_correction_micron = 2.5
        max_tip_correction_as = 20.0
        max_tilt_correction_as = 20.0

        # Modified
        # max_piston_correction_micron = .25
        # max_tip_correction_as = 2.0
        # max_tilt_correction_as = 2.0

        max_piston_correction_meters = max_piston_correction_micron * 1e-6
        max_tip_correction_radians = max_tip_correction_as * np.pi / (180 * 3600)
        max_tilt_correction_radians = max_tilt_correction_as * np.pi / (180 * 3600)


        # TODO: Refactor to extract tuples to np array and use _apply_ptt_displacements.
        for segment_id in range(self.num_apertures):

            # TODO: the following three lines need be generalized to nested lists.
            segment_piston_command = segments_ptt_commands[segment_id][0]

            if len(segments_ptt_commands[segment_id]) == 3:
                segment_tip_command = segments_ptt_commands[segment_id][1]
                segment_tilt_command = segments_ptt_commands[segment_id][2]

            else:
                segment_tip_command = 0.0
                segment_tilt_command = 0.0

            # segment_piston_command = ((segment_piston_command - (0.5)) * 2)
            # segment_tip_command = ((segment_tip_command - (0.5)) * 2)
            # segment_tilt_command = ((segment_tilt_command - (0.5)) * 2)

            segment_piston_command_meters = segment_piston_command * max_piston_correction_meters
            segment_tip_command_radians = segment_tip_command * max_tip_correction_radians
            segment_tilt_command_radians = segment_tilt_command * max_tilt_correction_radians


            # TODO: Externalize.
            direct_command = False

            if direct_command:

                piston_state = self.segment_baseline_dict[segment_id]["piston"] + segment_piston_command_meters
                tip_state = self.segment_baseline_dict[segment_id]["tip"] + segment_tip_command_radians
                tilt_state = self.segment_baseline_dict[segment_id]["tilt"] + segment_tilt_command_radians

            else:

                # First, get the ptt displacements in meters of of this segement...
                (segment_piston,
                 segment_tip,
                 segment_tilt) = self.segmented_mirror.get_segment_actuators(segment_id)

                piston_state = segment_piston + segment_piston_command_meters
                tip_state = segment_tip + segment_tip_command_radians
                tilt_state = segment_tilt + segment_tilt_command_radians

                # Enforce limits on incremental commmands with clip.
                piston_state = np.clip(
                    piston_state,
                    -max_piston_correction_meters + self.segment_baseline_dict[segment_id]["piston"],
                    max_piston_correction_meters + self.segment_baseline_dict[segment_id]["piston"])
                tip_state = np.clip(
                    tip_state,
                    -max_tip_correction_radians + self.segment_baseline_dict[segment_id]["tip"],
                    max_tip_correction_radians + self.segment_baseline_dict[segment_id]["tip"])
                tilt_state = np.clip(
                    tilt_state,
                    -max_tilt_correction_radians + self.segment_baseline_dict[segment_id]["tilt"],
                    max_tilt_correction_radians + self.segment_baseline_dict[segment_id]["tilt"])
                
                # TODO: this is wrong
                # piston_state = self.segment_baseline_dict[segment_id]["piston"] + piston_state
                # tip_state = self.segment_baseline_dict[segment_id]["tip"] + tip_state
                # tilt_state = self.segment_baseline_dict[segment_id]["tilt"] + tilt_state


                # if abs(piston_state) > abs(max_piston_correction_meters):
                    
                #     # If a command would exceed limits, it is ignored entirely.
                #     piston_state = segment_piston
                
                # if abs(tip_state) > abs(max_tip_correction_radians):

                #     # If a command would exceed limits, it is ignored entirely.
                #     tip_state = segment_tip

                # if abs(tilt_state) > abs(max_tilt_correction_radians):

                #     # If a command would exceed limits, it is ignored entirely.
                #     tilt_state = segment_tilt

            # Set the actuators in meter and radians.
            self.segmented_mirror.set_segment_actuators(
                segment_id,
                piston_state,
                tip_state,
                tilt_state
            )


    
class OptomechEnv(gym.Env):
    """
    Description:
        A distributed aperture telescope is tasked to observe an astrophysical 
        scene. This scene and the intervening atmosphere cause an at-aperture
        illuminance for each aperture. Each apertures reflects light onto an
        articulated secondary mirror. Finally, the light is split and then
        propogated to both a science camera and a wavefront sensor. The science
        camera produces a focal plane image, while the wavefront sensor
        produces a set of slopes. The agent is tasked with controlling the 
        optomechanical system, including the optomechanical support structure 
        actuators, the articulated secondary mirror actuators, and the
        defromable mirror actuators. Several reawrd functions are available, 
        and the control surfaces, scene, and atmosphere are all configurable.

        We adopt the convention that an environment step comprises a single
        occurence of the longest amonst the frame, control, and decision (i.e.,
        model inference) intervals. (Typically, this will be the decision 
        interval, but this is not guaranteed.) For instance, if the control,
        frame, and model inference intervals are assumed to be 4, 12, and 24 
        ms, respectively, then a single environment step will take 6 (24/4) 
        command matrices and produce 3 (12/4) focal plane images. These ratios, 
        along with the details of the focal plane, determine the action and 
        observation space shapes, and by extension the agent input and output
        shapes. The underlying optical system simulation will be evolved on a 
        time scale identical to that of the shortest amongst the frame,
        control, and model inference intervals. (In the example above, this 
        would be 4 ms.)

    Source:
        This environment corresponds to the Markov decision process described
        in the forthcoming work.

    Observation: 
        Type: Box(height, width)
        Each observation is a single frame of the focal plane image.
        
    Actions:

    Reward:
        Several reward functions are available. The default is the negative
        sum of the squared differences between the current and target focal
        plane images. Reward can be configured using the reward_function
        parameter.

    Starting State:
        Currently undefined.

    Episode Termination:
        Currently undefined.
    """
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self, **kwargs):

        # Set the seed.
        self.seed()

        # Store the keyword arguements.
        self.kwargs = kwargs
        self.uuid = uuid.uuid4()

        # Parse run configuration.
        self.report_time = kwargs['report_time']
        # self.render_mode = kwargs['render_mode']
        self.render_dpi = kwargs['render_dpi']
        self.record_env_state_info = kwargs['record_env_state_info']

        # Parse simulation parameters.
        self.render_frequency = kwargs['render_frequency']
        self.control_interval_ms = kwargs['control_interval_ms']
        self.frame_interval_ms = kwargs['frame_interval_ms']
        self.decision_interval_ms = kwargs['decision_interval_ms']
        self.ao_interval_ms = kwargs['ao_interval_ms']
        self.randomize_dm = kwargs['randomize_dm']
        self.reward_function = kwargs['reward_function']
        self.ao_loop_active = kwargs['ao_loop_active']

        self.command_tensioners = kwargs['command_tensioners']
        self.command_secondaries = kwargs['command_secondaries']
        self.command_dm = kwargs['command_dm']

        if self.command_dm or self.ao_loop_active:
            kwargs['model_ao'] = True
        else:

            kwargs['model_ao'] = False

        # TODO: Externalize these.
        self.microns_opd_per_actuator_bit = 0.00015
        self.stroke_count_limit = 20000
        # self.dm_gain = 0.9
        self.dm_gain = 0.6
        self.dm_leakage = 0.01
        # self.dm_leakage = 0.001

        print("==== Start: Initializing Environment ====")

        # Set the default values of the envrionment varaibles.
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        # TODO: Build a more robust state stroage structure.
        self.state_content = dict()
        self.state_content["dm_surfaces"] = list()
        self.state_content["atmos_layer_0_list"] = list()
        self.state_content["action_times"] = list()
        self.state_content["object_fields"] = list()
        self.state_content["pre_atmosphere_object_wavefronts"] = list()
        self.state_content["post_atmosphere_wavefronts"] = list()
        self.state_content["segmented_mirror_surfaces"] = list()
        self.state_content["pupil_wavefronts"] = list()
        self.state_content["post_dm_wavefronts"] = list()
        self.state_content["focal_plane_wavefronts"] = list()
        self.state_content["readout_images"] = list()
        self.state_content["instantaneous_psf"] = list()

        # Print the provided intervals
        print("Control interval: %s ms" % self.control_interval_ms)
        print("Frame interval: %s ms" % self.frame_interval_ms)
        print("Decision interval: %s ms" % self.decision_interval_ms)
        print("AO interval: %s ms" % self.ao_interval_ms)

        # Compute the number of commands per decision and frame, rounding up.
        self.commands_per_decision = math.ceil(
            self.decision_interval_ms / self.control_interval_ms
        )
        self.metadata['commands_per_decision'] = self.commands_per_decision
        self.commands_per_frame = math.ceil(
            self.frame_interval_ms / self.control_interval_ms
        )
        self.metadata['commands_per_frame'] = self.commands_per_frame
        self.frames_per_decision = math.ceil(
            self.decision_interval_ms / self.frame_interval_ms
        )
        self.metadata['frames_per_decision'] = self.frames_per_decision
        self.ao_steps_per_command = math.ceil(
            self.control_interval_ms/ self.ao_interval_ms
        )
        self.metadata['ao_steps_per_command'] = self.ao_steps_per_command

        self.ao_steps_per_frame = self.ao_steps_per_command * self.commands_per_frame 
        self.metadata['ao_steps_per_frame'] = self.ao_steps_per_frame

        print("Commands per decision: %s" % self.commands_per_decision)
        print("Commands per frame: %s" % self.commands_per_frame)
        print("AO steps per frame: %s" % self.ao_steps_per_frame)
        print("Frames per decision: %s" % self.frames_per_decision)


        # Build the optical system by passing in the kwargs.
        self.build_optical_system(**kwargs)

        # Reset the episode clock.
        self.episode_time_ms = 0.0

        # TODO: Externalize.
        self.command_tip_tilt = False

        # Build the command spaces and add them to a list.
        command_space_list = list()

        # Build the secondaries command space.
        if self.command_secondaries:

            # Secondaries can be piston-only, so we build a list for subspaces.
            ptt_space_list = list()
            
            # Build a piston space. We assume [-1., 1.] spaces for commands.
            # The physical units for command must be handled post-interface.
            self.secondary_max_displacement_micron = 1.0
            piston_space = spaces.Box(
                low=-self.secondary_max_displacement_micron,
                high=self.secondary_max_displacement_micron,
                shape=(1,),
                dtype=np.float32
            )
            ptt_space_list.append(piston_space)

            # If tip and tilt controls are modeled, add thier spaces.
            if self.command_tip_tilt:
            
                self.secondary_max_deflection_arcsec = 1.0
                tip_space = spaces.Box(
                    low=-self.secondary_max_deflection_arcsec,
                    high=self.secondary_max_deflection_arcsec,
                    shape=(1,),
                    dtype=np.float32
                )

                ptt_space_list.append(tip_space)
                tilt_space = spaces.Box(
                    low=-self.secondary_max_deflection_arcsec,
                    high=self.secondary_max_deflection_arcsec,
                    shape=(1,),
                    dtype=np.float32
                )

                ptt_space_list.append(tilt_space)   

            # Convert the list to a tuple and the tuple to a Tuple space.
            ptt_space = spaces.Tuple((tuple(ptt_space_list))) 

            # Create one tuple for each sequence element.
            secondaries_tuple = tuple([ptt_space] * self.optical_system.num_apertures)

            # Combine the tuple sequence into a final Gym space.
            secondaries_space = spaces.Tuple(secondaries_tuple)

            # Add the finalized secondaries command space to the system list.
            command_space_list.append(secondaries_space)

        if self.command_tensioners:

            self.tensioner_max_force = 1.0

            tensioner_space = spaces.Box(
                low=-self.tensioner_max_force,
                high=self.tensioner_max_force,
                shape=(1,),
                dtype=np.float32
            )

            tensioners_tuple = tuple([tensioner_space] * self.optical_system.num_tensioners)

            tensioners_space = spaces.Tuple(tensioners_tuple)
            command_space_list.append(tensioners_space)


        if self.command_dm:

            # Build the command grid, which is one-to-one with the action space.
            # TODO: In the active optics formulation, this needs to be ttp secondaries and tensioners.
            # TODO: Retain the ability to control any subset fo actuators.
            # self.actuator_command_grid = np.zeros(
            #     shape=(35, 35),
            #     dtype=np.int16
            # )
            # Define a symmetric action space.
            # action_shape = (self.commands_per_decision,
            #                 self.actuator_command_grid.shape[0],
            #                 self.actuator_command_grid.shape[1],)
            # self.action_space = spaces.Box(low=-self.stroke_count_limit,
            #                                high=self.stroke_count_limit,
            #                                shape=action_shape,
            #                                dtype=np.int16)

            # TODO: Externalize
            self.dm_stroke_micron = 1.0
            dm_actuator_space = spaces.Box(
                low=-self.dm_stroke_micron,
                high=self.dm_stroke_micron,
                shape=(1,),
                dtype=np.float32
            )

            dm_tuple = tuple([dm_actuator_space] * len(self.optical_system.dm.actuators))
            dm_space = spaces.Tuple(dm_tuple)

            command_space_list.append(dm_space)
        
        # TODO: Refactor to use spaces.Dict for easier integration w/ hardware.
        single_command_space = spaces.Tuple(tuple(command_space_list))
        
        self.dict_action_space = spaces.Tuple([single_command_space] * self.commands_per_decision)

        # Build a tree from the action space to enable translation.
        self.action_tree = self.build_tree_from_action_space(self.dict_action_space)
        self.action_space = self.flatten(self.dict_action_space,
                                         flat_space_low=-1.0,
                                         flat_space_high=1.0)
        
        # Define the zero action space.
        zero_command_space_list = list()
        
        if self.command_secondaries:

            zero_ptt_space_list = list()
            zero_piston_space = spaces.Box(
                low=0.0,
                high=0.0,
                shape=(1,),
                dtype=np.float32
            )

            zero_ptt_space_list.append(zero_piston_space)

            if self.command_tip_tilt:
            
                zero_tip_space = spaces.Box(
                    low=0.0,
                    high=0.0,
                    shape=(1,),
                    dtype=np.float32
                )

                zero_ptt_space_list.append(zero_tip_space)

                zero_tilt_space = spaces.Box(
                    low=0.0,
                    high=0.0,
                    shape=(1,),
                    dtype=np.float32
                )

                zero_ptt_space_list.append(zero_tilt_space)

            zero_ptt_space = spaces.Tuple((tuple(zero_ptt_space_list))) 

            zero_secondaries_tuple = tuple([zero_ptt_space] * self.optical_system.num_apertures)

            zero_secondaries_space = spaces.Tuple(zero_secondaries_tuple)

            zero_command_space_list.append(zero_secondaries_space)

        if self.command_tensioners:

            zero_tensioner_space = spaces.Box(
                low=0.0,
                high=0.0,
                shape=(1,),
                dtype=np.float32
            )

            zero_tensioners_tuple = tuple([zero_tensioner_space] * self.optical_system.num_tensioners)

            zero_tensioners_space = spaces.Tuple(zero_tensioners_tuple)

            zero_command_space_list.append(zero_tensioners_space)


        if self.command_dm:

            zero_dm_actuator_space = spaces.Box(
                low=0.0,
                high=0.0,
                shape=(len(self.optical_system.dm.actuators),),
                dtype=np.float32
            )

            zero_dm_tuple = tuple([zero_dm_actuator_space])

            zero_dm_space = spaces.Tuple(zero_dm_tuple)

            zero_command_space_list.append(zero_dm_space)
        
        zero_single_command_space = spaces.Tuple(
            tuple(zero_command_space_list)
        )

        self.zero_dict_action_space = spaces.Tuple([zero_single_command_space] * self.commands_per_decision)

        self.zero_action_space = self.flatten(self.zero_dict_action_space,
                                              flat_space_low=0.0,
                                              flat_space_high=0.0)

        # Compute the image shape.
        self.image_shape = (kwargs['focal_plane_image_size_pixels'],
                            kwargs['focal_plane_image_size_pixels'])

        # Define a 3D observation space.
        observation_shape = (self.frames_per_decision,
                             self.image_shape[0],
                             self.image_shape[1],)
        print("Observation shape: %s" % str(observation_shape))

        # TODO: Refactor to be int16.
        self.observation_space = spaces.Box(low=0.0,
                                            high=1.0,
                                            shape=observation_shape,
                                            dtype=np.float64)
        
        # TODO: Replace with environment-level state storage.
        self.state_content["wavelength"] = self.optical_system.wavelength

        print("==== End: Initializing Environment ====")


    def flatten(self, dict_space, flat_space_high=1.0, flat_space_low=0.0):

        flat_space = spaces.Box(
            low=flat_space_low,
            high=flat_space_high,
            shape=(self.get_vector_action_size(dict_space),),
            dtype=np.float32
        )

        return flat_space


    def flat_to_dict(self, flat_action, dict_space):

        dict_action = self.encode_action_space_from_vector(dict_space, flat_action)

        return dict_action


    def tuple_to_list(self, tup):
        if isinstance(tup, tuple):
            return [self.tuple_to_list(i) for i in tup]
        else:
            return tup


    def list_to_tuple(self, lst):
        if isinstance(lst, list):
            return tuple(self.list_to_tuple(i) for i in lst)
        else:
            return lst

    def build_tree_from_action_space(self, action_space):

        action_node = Node("action",
                        content="")

        linear_address = 0

        linear_to_tree_dict = {}


        # Iterate over each predictive command.
        for step_num, step in enumerate(action_space):

            step_node = Node(f"step_{step_num}",
                            parent=action_node,
                            content="")

            # Iterate over stages (e.g., secondaries, primary, DM) in the step.
            for stage_num, stage in enumerate(step):

                stage_node = Node(f"stage_{stage_num}",
                                parent=step_node,
                                content="")

                # Iterate over components (e.g., secondary, tensioner) in the stage.
                for component_num, component in enumerate(stage):

                    component_node = Node(f"component_{component_num}",
                                        parent=stage_node,
                                        content="")


                    if hasattr(component, '__iter__'):

                        # Iterate over commands (e.g., force, displacement) for the component.
                        for command_num, command in enumerate(component):



                            action_space_address= f"{step_num}_{stage_num}_{component_num}_{command_num}"
                            

                            command_node = Node(f"command_{command_num}",
                                                parent=component_node,
                                                content=command,
                                                action_space_address=action_space_address,
                                                linear_address=linear_address)

                            linear_to_tree_dict[linear_address] = action_space_address

                            linear_address += 1

                    else:


                        action_space_address= f"{step_num}_{stage_num}_{component_num}_{0}"

                        command_node = Node(f"command_0",
                                            parent=component_node,
                                            content=component,
                                            action_space_address=action_space_address,
                                            linear_address=linear_address)

                        linear_to_tree_dict[linear_address] = action_space_address

                        linear_address += 1

        action_node.num_leaf_nodes = linear_address
        action_node.linear_to_tree_dict = linear_to_tree_dict
        return action_node


    def encode_action_space_from_vector(self, action_space, action_vector):

        try:

            action_tree = self.action_tree

        except:

            raise Warning("No action tree found. Building tree from action space.")
            action_tree = self.build_tree_from_action_space(action_space)


        action_space_list = self.tuple_to_list(action_space.sample())

        for n, action_value in enumerate(action_vector):

            tree_address = action_tree.linear_to_tree_dict[n]

            # Convert the string to a list of integers
            indices = list(map(int, tree_address.split('_')))

            # Use the indices to assign a new value to the corresponding element in the nested list
            sublist = action_space_list
            for index in indices[:-1]:
                sublist = sublist[index]
            sublist[indices[-1]] = action_value

        action_space_tuple = self.list_to_tuple(action_space_list)

        return action_space_tuple


    def get_vector_action_size(self, hierarchical_action_space):

        try:

            action_space_tree = self.action_tree

        except:

            raise Warning("No action tree found. Building tree from action space.")

            action_space_tree = self.build_tree_from_action_space(hierarchical_action_space)

        vector_action_size = action_space_tree.num_leaf_nodes

        return vector_action_size


    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)

        return [seed]


    def reset(self, seed=None, options=None):

        print("=== Start: Reset Environment ===")
        full_reset = True
        if full_reset: 

            # Set the initial state. This is the first thing called in an episode.
            print("Instantiating a New Optical System")

            self.build_optical_system(**self.kwargs)

            if self.command_dm or self.ao_loop_active:
                # Calibrate the DM interaction matrix.
                self.optical_system.calibrate_dm_interaction_matrix(self.uuid)
                rcond = 1e-3
                self.reconstruction_matrix = hcipy.inverse_tikhonov(
                    self.optical_system.interaction_matrix.transformation_matrix,
                    rcond=rcond)

                self.episode_time_ms = 0.0
        
            print("Populating Initial Action")

 
        # Initialize natural structural differential motion.
        if self.kwargs['init_differential_motion']:
            print("Initializing differential motion.")
            self.optical_system._init_natural_diff_motion()

        self.optical_system._store_baseline_segment_displacements()

        print("Populating Initial State")
        for _ in range(self.frames_per_decision):

            (initial_state, _, _, _, info) = self.step(
                action=self.zero_action_space.sample(),
                noisy_command=False,
                reset=True
            )

        self.state = initial_state
        self.steps_beyond_done = None

        print("=== End: Reset Environment ===")
        return (np.array(self.state), info)
    
    def save_state(self):

        # We have now completed the substep; store the state variables.
        if self.report_time:
            deepcopy_start = time.time()

        if self.optical_system.model_ao:

            self.state_content["dm_surfaces"].append(
                copy.deepcopy(self.optical_system.dm.surface)
            )

        # TODO: Add support for saving the rest of the atmosphere layers.
        if len(self.optical_system.atmosphere_layers) > 0:

            deepcopy_atmosphere_layers = copy.deepcopy(
                self.optical_system.atmosphere_layers[0]
            )
            self.state_content["atmos_layer_0_list"].append(
                copy.deepcopy(deepcopy_atmosphere_layers)
            )
        else:
            self.state_content["atmos_layer_0_list"].append(None)

        deepcopy_object_plane = copy.deepcopy(self.optical_system.object_plane)
        self.state_content["object_fields"].append(
            deepcopy_object_plane
        )

        deepcopy_pre_atmosphere_object_wavefront = copy.deepcopy(self.optical_system.pre_atmosphere_object_wavefront)
        self.state_content["pre_atmosphere_object_wavefronts"].append(
            deepcopy_pre_atmosphere_object_wavefront
        )
        
        deepcopy_post_atmosphere_wavefront = copy.deepcopy(self.optical_system.post_atmosphere_wavefront)
        self.state_content["post_atmosphere_wavefronts"].append(
            deepcopy_post_atmosphere_wavefront
        )
        
        deepcopy_segmented_mirror_surface = copy.deepcopy(self.optical_system.segmented_mirror.surface)
        self.state_content["segmented_mirror_surfaces"].append(
            deepcopy_segmented_mirror_surface
        )
        
        deepcopy_pupil_wavefront = copy.deepcopy(self.optical_system.pupil_wavefront)
        self.state_content["pupil_wavefronts"].append(
            deepcopy_pupil_wavefront
        )

        deepcopy_post_dm_wavefront = copy.deepcopy(self.optical_system.post_dm_wavefront)
        self.state_content["post_dm_wavefronts"].append(
            deepcopy_post_dm_wavefront
        )
        
        deepcopy_focal_plane_wavefront = copy.deepcopy(self.optical_system.focal_plane_wavefront)
        self.state_content["focal_plane_wavefronts"].append(
            deepcopy_focal_plane_wavefront
        )

        deepcopy_instantaneous_psf = copy.deepcopy(self.optical_system.instantaneous_psf)
        self.state_content["instantaneous_psf"].append(
            deepcopy_instantaneous_psf
        )
        
        deepcopy_science_readout_raster = copy.deepcopy(self.science_readout_raster)
        self.state_content["readout_images"].append(
            deepcopy_science_readout_raster
        )

        if self.report_time:
            print("- Deepcopy time:   %.6f" %
                    (time.time() - deepcopy_start))

    def step(self,
             action,
             noisy_command=False,
             reset=False):
        
        self.report_name = False
        if self.report_name:
            print("Step from Environment %s" % self.uuid)


        if self.report_time:
            step_time = time.time()


        # First, ensure the step action is valid.
        assert self.action_space.contains(action), \
               "%r (%s) invalid"%(action, type(action))

        # Clear the custom state content for population.
        # TODO: encapsulate this mess...
        if self.record_env_state_info:
            self.state_content["dm_surfaces"] = list()
            self.state_content["atmos_layer_0_list"] = list()
            self.state_content["action_times"] = list()
            self.state_content["object_fields"] = list()
            self.state_content["pre_atmosphere_object_wavefronts"] = list()
            self.state_content["post_atmosphere_wavefronts"] = list()
            self.state_content["segmented_mirror_surfaces"] = list()
            self.state_content["pupil_wavefronts"] = list()
            self.state_content["post_dm_wavefronts"] = list()
            self.state_content["focal_plane_wavefronts"] = list()
            self.state_content["readout_images"] = list()
            self.state_content["instantaneous_psf"] = list()
            self.state_content["shwfs_slopes"] = list()

        # Update the current action to be the provided action.

        # TODO: Convert vector action to tree action here.
        self.action = self.flat_to_dict(action, self.dict_action_space)
        
        # TODO: Replace this with a static property.
        self.focal_plane_images = list()
        self.shwfs_slopes_list = list()

        # Run the step simulation loop.
        for frame_num in range(self.frames_per_decision):

            # Create a blank frame for manual integration.
            frame = np.zeros(self.image_shape, dtype=np.float64)

            # Iterate over each command, applying it and integrating the frame.
            for command_num in range(self.commands_per_frame):

                self.episode_time_ms += self.control_interval_ms

                # Evolve the atmosphere to the current time.
                atmospere_evolution_start = time.time()
                self.optical_system.evolve_atmosphere_to(self.episode_time_ms)
                if self.report_time:
                    print("- Atmosphere time: %.6f" % (time.time() - atmospere_evolution_start))

                # Get the commanded actuations.
                command = self.action[command_num]
                    
                if self.command_tensioners:

                    tensioner_commands = command[1]
                    self.optical_system.command_tensioners(tensioner_commands)

                if self.command_secondaries:

                    secondaries_commands = command[0]
                    self.optical_system.command_secondaries(secondaries_commands)

                if self.command_dm:
                    
                    dm_command = command[2]

                    self.optical_system.command_dm(dm_command)

                # Compute the number of seconds of integration per AO loop step.
                frame_interval_seconds = self.frame_interval_ms / 1000.0
                integration_seconds = frame_interval_seconds / self.ao_steps_per_frame

                # Iterate, simulating an AO loop.
                for ao_step_num in range(self.ao_steps_per_command):

                    if self.report_time:
                        simulation_time_start = time.time()
                
                    # Simulate the entire optical system, updating its state.
                    self.optical_system.simulate()

                    if self.report_time:
                        print("-- Simulation Step time: %.6f" % (time.time() - simulation_time_start))
                
                    if self.report_time:
                        ao_step_time_start = time.time()

                    # If the AO loop is active.
                    if self.ao_loop_active and not reset:

                        # Measure the wavefront using the SHWFS.
                        shwfs_readout_vector = self.optical_system.get_shwfs_frame(
                            integration_seconds=integration_seconds
                        )
                        
                        # Compute the correction slopes for the DM.
                        shwfs_slopes = self.optical_system.shwfse.estimate([shwfs_readout_vector + 1e-10])
                        shwfs_slopes -= self.optical_system.reference_slopes
                        self.shwfs_slopes = shwfs_slopes.ravel()
                        self.shwfs_slopes_list.append(self.shwfs_slopes)

                        # Perform wavefront compensation by setting the DM actuators.
                        self.optical_system.dm.actuators = (1 - self.dm_leakage) * self.optical_system.dm.actuators - self.dm_gain * self.reconstruction_matrix.dot(self.shwfs_slopes)

                        self.microns_opd_per_actuator_bit = 0.00015
                        self.stroke_count_limit = 20000

                        stroke_limit = self.microns_opd_per_actuator_bit * self.stroke_count_limit * 1e-6 / 2
                        
                        # Finally, clip actuator values to the stroke limit.
                        self.optical_system.dm.actuators = np.clip(
                            self.optical_system.dm.actuators,
                            -stroke_limit,
                            stroke_limit
                        )

                    if self.report_time:
                        print("-- AO Step time: %.6f" % (time.time() - ao_step_time_start))

                    # Get a fractional science frame.
                    science_readout_vector = self.optical_system.get_science_frame(
                        integration_seconds=integration_seconds
                    )
                    self.science_readout_raster = np.reshape(science_readout_vector,
                                                             self.image_shape)

                    # Note: This step accumulates the partial readout rasters,
                    # in effect manually integrating them outside of HCIPy.
                    frame += self.science_readout_raster

                    # Deepcopy the environment state so that it can be stored later.
                    if self.record_env_state_info and not reset:
                        self.save_state()
                    
            # Finally, append this frame to the stack of focal plane images.
            self.focal_plane_images.append(frame)

        # Encode the frames as 256 ** 2

        # Set the state to focal plane image.
        self.state = self.focal_plane_images

        if self.reward_function == "strehl":

            strehls = list()
        
            for focal_plane_image in self.focal_plane_images:

                # reshape to image
                # test_perfect_image = np.reshape(self.optical_system.perfect_image, self.image_shape)
                # plt.imshow(np.log(test_perfect_image / np.max(test_perfect_image)))
                # plt.colorbar()
                # plt.show()
                # die

                strehls.append(hcipy.metrics.get_strehl_from_focal(
                    focal_plane_image.flatten() / np.max(focal_plane_image),
                    self.optical_system.perfect_image / np.max(self.optical_system.perfect_image)
                ))

            reward = np.mean(strehls)
            
        elif self.reward_function == "negastrehl":

            strehls = list()
        
            for focal_plane_image in self.focal_plane_images:

                strehls.append(hcipy.metrics.get_strehl_from_focal(
                    focal_plane_image.flatten() / np.max(focal_plane_image),
                    self.optical_system.perfect_image / np.max(self.optical_system.perfect_image)
                ))

            reward = np.mean(strehls) - 1.0

        elif self.reward_function == "strehl_closed":

            strehls = list()
        
            for focal_plane_image in self.focal_plane_images:

                strehls.append(hcipy.metrics.get_strehl_from_focal(
                    focal_plane_image.flatten() / np.max(focal_plane_image),
                    self.optical_system.perfect_image / np.max(self.optical_system.perfect_image)
                ))

            
            reward = 1.0 if np.mean(strehls) >= 0.8 else 0.0

        elif self.reward_function == "ao_rms_slope":

            reward = 0.0

            if self.ao_loop_active:
                
                for shwfs_slopes in self.shwfs_slopes_list:

                    # TODO: replace the numerator with the max inverse rms slope.

                    reward += 1 / np.sqrt(np.mean(shwfs_slopes ** 2))

            else:

                reward = 0.0

        elif self.reward_function == "norm_ao_rms_slope":

            reward = 0.0

            if self.ao_loop_active:
                
                for shwfs_slopes in self.shwfs_slopes_list:

                    # TODO: replace the numerator with the max inverse rms slope.

                    reward += 1 / np.sqrt(np.mean(shwfs_slopes ** 2))

                reward = reward / 1e7

            else:

                reward = 0.0

        elif self.reward_function == "ao_closed":

            inverse_slope_threshold = 2e6

            invers_slope_rms = 0

            reward = 0

            if self.ao_loop_active:
                
                for shwfs_slopes in self.shwfs_slopes_list:

                    # TODO: replace the numerator with the max inverse rms slope.
                    invers_slope_rms += 1 / np.sqrt(np.mean(shwfs_slopes ** 2))

                if invers_slope_rms >= inverse_slope_threshold: 

                    reward = 1.0

                else:

                    reward = 0.0

            else:

                reward = 0.0

        elif self.reward_function == "negative_intensity":

            print(np.sum(self.state))

            reward = -1* np.sum(self.state)

        elif self.reward_function == "inverse_intensity":

            print(np.sum(self.state))

            reward = 1 / np.sum(self.state)

        else:

            raise ValueError("reward_function must be specified.")

        # TODO: compute_terminated()
        terminated = False
        
        # TODO: compute_truncated()
        truncated = False

        # Populate the information dictionary for this step.
        info = dict()
        
        if self.record_env_state_info:

            info["state_content"] = self.state_content
            info["state"] = np.array(self.state)

        if self.report_time:
            print("Step time: %.6f" % (time.time() - step_time))

        # TODO: Externalize
        normalize_state = True

        if normalize_state:

            raw_state = self.state

            raw_state_min = np.min(raw_state)
            zero_min_state = raw_state - raw_state_min
            zero_min_state_max = np.max(zero_min_state)
            normalized_state = zero_min_state / zero_min_state_max

            self.state = normalized_state

        return np.array(self.state), reward, terminated, truncated, info

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def build_optical_system(self, **kwargs):

        self.optical_system = OpticalSystem(**kwargs)

        return