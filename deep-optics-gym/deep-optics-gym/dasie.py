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

        self.report_time = kwargs["report_time"]
        self.microns_opd_per_actuator_bit = 0.00015
        self.num_apertures = 15

        self.num_tensioners = kwargs["num_tensioners"]

        # Parameters for the pupil function
        # focal_length = 200.0 # m
        focal_length = 200.0 # m
        pupil_diameter = 3.6 # m
        elf_segment_centroid_diameter = 2.5 # m
        
        # Parameters for the optical simulation.
        num_pupil_grid_simulation_pixels = kwargs['focal_plane_image_size_pixels']
        # self.wavelength = 763e-9
        self.wavelength = 1000e-9
        aperture_type = kwargs['aperture_type']
        oversampling_factor = 8

        # Parameters for the instrument. 
        # TODO: Make use of these.
        flat_field = 0.00001
        dark_current_rate = 1

        # Parameters for the atmosphere. Caution: better seeing = longer runs.
        # TODO: Externalize.
        seeing = 0.6 # arcsec @ 500nm (convention)
        outer_scale = 40 # meter
        tau0 = 1.0 / 30.0 # seconds (30 hz)

        self.simulate_differential_motion = kwargs['simulate_differential_motion']
        self.init_differential_motion = kwargs['init_differential_motion']

        # Parameters for the structual wind response model.
        # TODO: Externalize.
        initial_ground_wind_speed_mps = 16.7
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
        # TODO: Externalize.
        kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-3
        focal_plane_extent_metres = kwargs['focal_plane_image_size_meters']

        airy_extent_radians = 1.22 * self.wavelength / pupil_diameter
        airy_extent_meters = airy_extent_radians * focal_length
        focal_plane_pixel_extent_meters = focal_plane_extent_metres / num_focal_grid_pixels
        sampling = airy_extent_meters / focal_plane_pixel_extent_meters



        focal_plane_resolution_element = self.wavelength * focal_length / pupil_diameter
        focal_plane_pixels_per_meter = num_focal_grid_pixels / focal_plane_extent_metres
        focal_plane_pixel_extent_meters = focal_plane_extent_metres /  num_focal_grid_pixels
        self.ifov = 206265 / focal_length * focal_plane_pixel_extent_meters


        # sampling: The number of pixels per resolution element (= lambda f / D).
        # sampling = focal_plane_resolution_element / focal_plane_pixel_extent_meters
        # num_airy: The spatial extent of the grid in radius in resolution elements (= lambda f / D).
        num_airy = num_focal_grid_pixels / (2 * sampling)



        # Initialize an empty structual interaction matrix.
        self.interaction_matrix = None
        interaction_size = 1
        self._optomech_encoder = np.random.rand(self.num_tensioners, interaction_size)
        self._optomech_decoder = np.random.rand(interaction_size, self.num_apertures * 3)


        # Log computed values to aid in debugging.
        print("num_focal_grid_pixels: %s" % num_focal_grid_pixels)
        print("focal_plane_extent_metres: %s" % focal_plane_extent_metres)
        print("focal_plane_pixel_extent_meters: %s" % focal_plane_pixel_extent_meters)
        print("focal_plane_resolution_element: %s" % focal_plane_resolution_element)
        print("focal_plane_pixels_per_meter: %s" % focal_plane_pixels_per_meter)
        print("num_airy (The spatial extent of the grid in radius in resolution elements): %s" % num_airy)
        print("sampling (number of pixels per resolution element): %s" % sampling)
        print("ifov (arcsec/pixel): %s" % self.ifov)

        # Build the object plane for this system.
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
        self.pupil_grid = hcipy.make_pupil_grid(
            dims=num_pupil_grid_simulation_pixels,
            diameter=pupil_diameter
        )

        # Initialize a list to store the atmosphere layers.
        self.atmosphere_layers = list()

        # Add the atmosphere layers.
        for layer_num in range(kwargs['num_atmosphere_layers']):

            layer = hcipy.InfiniteAtmosphericLayer(self.pupil_grid,
                                                   Cn_squared,
                                                   outer_scale,
                                                   velocity)
            
            self.atmosphere_layers.append(layer)
        
        # Make the simulation grid for the focal plane.
        focal_grid = hcipy.make_focal_grid(
            sampling,
            num_airy,
            spatial_resolution=self.wavelength * focal_length / pupil_diameter,
        )
        focal_grid = focal_grid.shifted(focal_grid.delta / 2)

        # Instantiate a Fraunhofer propagator from the pupil to focal plane.
        self.pupil_to_focal_propagator = hcipy.FraunhoferPropagator(
            self.pupil_grid,
            focal_grid, 
            focal_length
        )

        # Build the selected aperture; elf is the default.
        # TODO: Someday we should generalize this to accept any HCIPy aperture.
        if aperture_type == "elf":

            # Instantiate a segmented aperture.
            aperture, segments = self.make_elf_aperture(
                pupil_diameter=elf_segment_centroid_diameter,
                num_apertures=self.num_apertures,
                segment_diameter=0.5,
                return_segments=True,
            )

            # Evaluate the aperture, initializing them.
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

            # pupil_grid = hcipy.make_pupil_grid(
            #     256,
            #     diameter=pupil_diameter
            # )
            # focal_grid = hcipy.make_focal_grid(
            #     q=sampling,
            #     num_airy=num_airy,
            #     spatial_resolution=self.wavelength * focal_length / pupil_diameter,
            # )
            # prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)

            # pupil_grid = hcipy.make_pupil_grid(256)
            # wavefront = hcipy.Wavefront(aperture)
            # focal_grid = hcipy.make_focal_grid(q=8, num_airy=16)
            # prop = hcipy.FraunhoferPropagator(pupil_grid, focal_grid)
            # focal_image_thiers = prop.forward(wavefront)

            # fig = plt.figure()
            # plt.subplot(1, 2, 1)
            # plt.title('focal_image_thiers')
            # hcipy.imshow_field(np.log10(focal_image_thiers.intensity / focal_image_thiers.intensity.max()), vmin=-5)
            # plt.xlabel('Focal plane distance [$\lambda/D$]')
            # plt.ylabel('Focal plane distance [$\lambda/D$]')
            # plt.colorbar()
        
            # plt.subplot(1, 2, 2)
            # plt.title('focal_image_ours')
            # hcipy.imshow_field(np.log10(focal_image_ours.intensity / focal_image_ours.intensity.max()), vmin=-5)
            # plt.xlabel('Focal plane distance [$\lambda/D$]')
            # plt.ylabel('Focal plane distance [$\lambda/D$]')
            # plt.colorbar()
            # plt.show()
            # die

            self.perfect_image = perfect_image.intensity
    

            # Store the baseline segment displacements.
            self._store_baseline_segment_displacements()

        elif aperture_type == "circular":

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

            # Store the baseline segment displacements.
            self._store_baseline_segment_displacements()

        else:

            # Note: if you add aperture types, update this exception.
            raise NotImplementedError(
                "aperture_type was %s, but only 'elf' and 'circular' are \
                implemented." % aperture_type)
        
        # TODO: Externalize and modularize for other WFS types, including none.
        # Instantiate a Shack-Hartmann wavefront sensor.
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

        # TODO: Remove?
        # Build a DM on the pupil grid.
        # self.dm_influence_functions = hcipy.make_gaussian_influence_functions(
        #     self.pupil_grid,
        #     num_actuators_across_pupil=35,
        #     actuator_spacing=pupil_diameter / 35
        # )
        # self.dm = hcipy.DeformableMirror(self.dm_influence_functions)

        # Initialize natural structural differential motion.
        if self.init_differential_motion:
            print("Initializing differential motion.")
            self._init_natural_diff_motion()
            
        # Finally, make a camera.
        # Note: The camera is noiseless here; we add noise in the Env step().
        self.camera = hcipy.NoiselessDetector(focal_grid)

    

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


    def command_dm(self, command_grid):

        meters_opd_per_actuator_bit = self.microns_opd_per_actuator_bit * 1e-6
        command_vector = command_grid.flatten().astype(np.float64)
        self.dm.actuators += meters_opd_per_actuator_bit * command_vector

        return
    

    def simulate(self):


        # Chain together the wavefronts using the optical elements to produce a frame.

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

        dm_forawrd_start_time = time.time()
        # Propagate the wavefront from the segmented mirror through the DM.
        # Note: counter-intuitively, the DM must be re-applied after changes.
        self.post_dm_wavefront = self.dm.forward(self.pupil_wavefront)

        if self.report_time:

            print("--- DM Forward time: %0.6f" % (time.time() - dm_forawrd_start_time))
      
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
    

    def calibrate_dm_interaction_matrix(self):

        probe_amp = 0.01 * self.wavelength
        response_matrix = list()

        print("Calibrating DM.")

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


    def randomize_dm(self):

            self.dm.actuators = np.random.randn(len(self.dm.actuators)) / (np.arange(len(self.dm.actuators)) + 10)
            self.dm.actuators *= 0.3 * self.wavelength / np.std(self.dm.surface)
            # self.dm.flatten()
            # self.dm.random(1e-6)
    

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

            # self.instantaneous_psf
            # print("self.instantaneous_psf %3.16f" % np.std(self.instantaneous_psf))

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

        # Debug
        optomech_ptt_displacements = np.zeros((self.num_apertures, 3))


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
        # wind_ptt_displacements[:, 0] *= wind_diff_motion_piston_micron_std * 1e-6
        # # Sample displacement in radians.
        # wind_ptt_displacements[:, 1] *= wind_diff_motion_tip_arcsec_std * np.pi / (180 * 3600)
        # wind_ptt_displacements[:, 2] *= wind_diff_motion_tilt_arcsec_std * np.pi / (180 * 3600)
        self._apply_ptt_displacements(wind_ptt_displacements)
        
        # # Compute and apply the temperature displacments.
        # self.ground_temp_ms_sampled_std_mps
        # self.ground_temp_degcel
        # # TODO: Compute these values. Need help from Tim and Ye.
        # temp_ptt_displacements = np.random.randn(self.num_apertures, 3)
        # # Sample displacement in meters.
        # # TODO: This will always be 0.0 for now.
        # temp_ptt_displacements[:, 0] *= self.ground_temp_ms_sampled_std_mps * 1e-6
        # # Sample displacement in radians.
        # temp_ptt_displacements[:, 1] *= self.ground_temp_ms_sampled_std_mps * np.pi / (180 * 3600)
        # temp_ptt_displacements[:, 2] *= self.ground_temp_ms_sampled_std_mps * np.pi / (180 * 3600)
        # self._apply_ptt_displacements(temp_ptt_displacements)

        # # Compute and apply the gravity displacments.
        # self.gravity_normal_ms_sampled_std_mps
        # self.gravity_normal_deg
        # # TODO: Compute these values. Need help from Tim and Ye.
        # gravity_ptt_displacements = np.random.randn(self.num_apertures, 3)
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

        # Compute and apply the wind displacments.
        self.ground_wind_speed_ms_sampled_std_mps
        self.ground_wind_speed_mps
        # TODO: Compute these values. Need help from Tim and Ye.
        wind_diff_motion_piston_micron_std = 1.0
        wind_diff_motion_tip_arcsec_std = 0.25
        wind_diff_motion_tilt_arcsec_std = 0.25
        wind_ptt_displacements = np.random.randn(self.num_apertures, 3)
        # Sample displacement in meters.
        # wind_ptt_displacements[:, 0] *= wind_diff_motion_piston_micron_std * 1e-6
        # # Sample displacement in radians.
        # wind_ptt_displacements[:, 1] *= wind_diff_motion_tip_arcsec_std * np.pi / (180 * 3600)
        # wind_ptt_displacements[:, 2] *= wind_diff_motion_tilt_arcsec_std  * np.pi / (180 * 3600)
        self._apply_ptt_displacements(wind_ptt_displacements)
        
        # Compute and apply the temperature displacments.
        self.ground_temp_ms_sampled_std_mps
        self.ground_temp_degcel
        # TODO: Compute these values. Need help from Tim and Ye.
        temp_diff_motion_piston_micron_std = 0.0
        temp_diff_motion_tip_arcsec_std = 0.0
        temp_diff_motion_tilt_arcsec_std = 0.0
        temp_ptt_displacements = np.random.randn(self.num_apertures, 3)
        # Sample displacement in meters.
        # temp_ptt_displacements[:, 0] *= temp_diff_motion_piston_micron_std * 1e-6
        # # Sample displacement in radians.
        # temp_ptt_displacements[:, 1] *= temp_diff_motion_tip_arcsec_std  * np.pi / (180 * 3600)
        # temp_ptt_displacements[:, 2] *= temp_diff_motion_tilt_arcsec_std  * np.pi / (180 * 3600)
        self._apply_ptt_displacements(temp_ptt_displacements)

        # Compute and apply the gravity displacments.
        self.gravity_normal_ms_sampled_std_mps
        self.gravity_normal_deg
        # TODO: Compute these values. Need help from Tim and Ye.
        gravity_diff_motion_piston_micron_std = 300.0
        gravity_diff_motion_tip_arcsec_std = 15.0
        gravity_diff_motion_tilt_arcsec_std = 15.0
        gravity_ptt_displacements = np.random.randn(self.num_apertures, 3)
        # Sample displacement in meters.
        # gravity_ptt_displacements[:, 0] *= gravity_diff_motion_piston_micron_std * 1e-6
        # # Sample displacement in radians.
        # gravity_ptt_displacements[:, 1] *= gravity_diff_motion_tip_arcsec_std * np.pi / (180 * 3600) 
        # gravity_ptt_displacements[:, 2] *= gravity_diff_motion_tilt_arcsec_std * np.pi / (180 * 3600)
        self._apply_ptt_displacements(gravity_ptt_displacements)
    
  
    def _apply_ptt_displacements(self, ptt_displacements):
        """
        Apply the provided incremental PTT displacements to the segmented
        mirror.

        Parameters
        ----------
        ptt_displacements : np.ndarray
            An array of shape (num_apertures, 3) containing the commanded
            PTT displacements to be applied to each segment. These are scaled
            from 0.0 to 1.0, and must be converted to physical units here.


        """

        segments_ptt_commands = ptt_displacements

        max_piston_correction_micron = 3000.0
        max_tip_correction_as = 2000.0
        max_tilt_correction_as = 2000.0

        max_piston_correction_meters = max_piston_correction_micron * 1e-6
        max_tip_correction_radians = max_tip_correction_as * np.pi / (180 * 3600)
        max_tilt_correction_radians = max_tilt_correction_as * np.pi / (180 * 3600)

        # Iterate over each segment, applying the provided displacements.
        for segment_id in range(self.num_apertures):

            # TODO: the following three lines need be generalized to nested lists.
            segment_piston_command = segments_ptt_commands[segment_id, 0]
            segment_tip_command = segments_ptt_commands[segment_id, 1]
            segment_tilt_command = segments_ptt_commands[segment_id, 2]

            # segment_piston_command = ((segment_piston_command - (0.5)) * 2)
            # segment_tip_command = ((segment_tip_command - (0.5)) * 2)
            # segment_tilt_command = ((segment_tilt_command - (0.5)) * 2)


            segment_piston_command_meters = segment_piston_command * max_piston_correction_meters
            segment_tip_command_radians = segment_tip_command * max_tip_correction_radians
            segment_tilt_command_radians = segment_tilt_command * max_tilt_correction_radians


            # TODO: Externalize.
            direct_command = True

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

                # Enforce limits on incremental commmands.
                if abs(piston_state) > abs(max_piston_correction_meters):
                    
                    # If a command would exceed limits, it is ignored entirely.
                    piston_state = segment_piston
                
                if abs(tip_state) > abs(max_tip_correction_radians):

                    # If a command would exceed limits, it is ignored entirely.
                    tip_state = segment_tip

                if abs(tilt_state) > abs(max_tilt_correction_radians):

                    # If a command would exceed limits, it is ignored entirely.
                    tilt_state = segment_tilt


            # Set the actuators in meter and radians.
            self.segmented_mirror.set_segment_actuators(
                segment_id,
                piston_state,
                tip_state,
                tilt_state
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
    
        max_piston_correction_micron = 2.5
        max_tip_correction_as = 20.0
        max_tilt_correction_as = 20.0


        max_piston_correction_meters = max_piston_correction_micron * 1e-6
        max_tip_correction_radians = max_tip_correction_as * np.pi / (180 * 3600)
        max_tilt_correction_radians = max_tilt_correction_as * np.pi / (180 * 3600)


        # TODO: Refactor to extract tuples to np array and use _apply_ptt_displacements.
        for segment_id in range(self.num_apertures):

            # TODO: the following three lines need be generalized to nested lists.
            segment_piston_command = segments_ptt_commands[segment_id][0]
            segment_tip_command = segments_ptt_commands[segment_id][1]
            segment_tilt_command = segments_ptt_commands[segment_id][2]

            # segment_piston_command = ((segment_piston_command - (0.5)) * 2)
            # segment_tip_command = ((segment_tip_command - (0.5)) * 2)
            # segment_tilt_command = ((segment_tilt_command - (0.5)) * 2)

            segment_piston_command_meters = segment_piston_command * max_piston_correction_meters
            segment_tip_command_radians = segment_tip_command * max_tip_correction_radians
            segment_tilt_command_radians = segment_tilt_command * max_tilt_correction_radians


            # TODO: Externalize.
            direct_command = True

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

                # Enforce limits on incremental commmands.
                if abs(piston_state) > abs(max_piston_correction_meters):
                    
                    # If a command would exceed limits, it is ignored entirely.
                    piston_state = segment_piston
                
                if abs(tip_state) > abs(max_tip_correction_radians):

                    # If a command would exceed limits, it is ignored entirely.
                    tip_state = segment_tip

                if abs(tilt_state) > abs(max_tilt_correction_radians):

                    # If a command would exceed limits, it is ignored entirely.
                    tilt_state = segment_tilt


            # Set the actuators in meter and radians.
            self.segmented_mirror.set_segment_actuators(
                segment_id,
                piston_state,
                tip_state,
                tilt_state
            ) 



    
class DasieEnv(gym.Env):
    """
    Description:
        A distributed aperture telescope is tasked to observe an extended space
        object. This object and the intervening atmosphere cause an at-aperture
        illuminance for each aperture. These apertures reflect light onto a
        deformable secondary mirror, which is actuated by a grid of actuators.

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
        Type: N x N int16 ndarray

    Reward:
        Currently undefined. Eventually computed from SNIIRS gains.

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
        # TODO: Externalize.
        self.microns_opd_per_actuator_bit = 0.00015
        self.stroke_count_limit = 20000
        self.dm_gain = 0.6
        self.dm_leakage = 0.01

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

        # Create a dict to hold the some hidden state content.
        self.build_optical_system(**kwargs)

        self.episode_time_ms = 0.0

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

        # FEATURE: Redefining to allow for better DM calibration.
        # num_dm_modes = 500
        # self.actuator_command_grid = np.zeros(
        #     shape=(num_dm_modes),
        #     dtype=np.float32
        # )
        # # FEATURE: redefining action space to match mode-based DM.
        # action_shape = (self.commands_per_decision,
        #                 num_dm_modes)
        # # FEATURE: redefining action space to match mode-based DM; replace stroke limits.
        # self.action_space = spaces.Box(low=-self.stroke_count_limit,
        #                                high=self.stroke_count_limit,
        #                                shape=action_shape,
        #                                dtype=np.float32)
        
        # # FEATURE: reconfiguring action space to optomech
        # num_tensioners = 16
        # self.optical_system.num_apertures

        # action_shape = (self.commands_per_decision,
        #                 ((1, num_tensioners),
        #                  (3, self.optical_system.num_apertures)
        #                 ))
        
        self.secondary_max_displacement_micron = 1.0
        piston_space = spaces.Box(
            low=-self.secondary_max_displacement_micron,
            high=self.secondary_max_displacement_micron,
            shape=(1,),
            dtype=np.float32
        )
        
        self.secondary_max_deflection_arcsec = 1.0
        tip_space = spaces.Box(
            low=-self.secondary_max_deflection_arcsec,
            high=self.secondary_max_deflection_arcsec,
            shape=(1,),
            dtype=np.float32
        )
        tilt_space = spaces.Box(
            low=-self.secondary_max_deflection_arcsec,
            high=self.secondary_max_deflection_arcsec,
            shape=(1,),
            dtype=np.float32
        )

        ptt_space = spaces.Tuple((piston_space, tip_space, tilt_space)) 

        secondaries_tuple = tuple([ptt_space] * self.optical_system.num_apertures)

        secondaries_space = spaces.Tuple(secondaries_tuple)

        self.tensioner_max_force = 1.0

        tensioner_space = spaces.Box(
            low=-self.tensioner_max_force,
            high=self.tensioner_max_force,
            shape=(1,),
            dtype=np.float32
        )

        tensioners_tuple = tuple([tensioner_space] * self.optical_system.num_tensioners)

        tensioners_space = spaces.Tuple(tensioners_tuple)
        
        single_command_space = spaces.Tuple((secondaries_space,
                                             tensioners_space))
        
        self.dict_action_space = spaces.Tuple([single_command_space] * self.commands_per_decision)

        # Build a tree from the action space to enable translation.
        self.action_tree = self.build_tree_from_action_space(self.dict_action_space)

        self.action_space = self.flatten(self.dict_action_space,
                                         flat_space_low=-1.0,
                                         flat_space_high=1.0)

        zero_piston_space = spaces.Box(
            low=0.0,
            high=0.0,
            shape=(1,),
            dtype=np.float32
        )
        
        zero_tip_space = spaces.Box(
            low=0.0,
            high=0.0,
            shape=(1,),
            dtype=np.float32
        )

        zero_tilt_space = spaces.Box(
            low=0.0,
            high=0.0,
            shape=(1,),
            dtype=np.float32
        )

        zero_ptt_space = spaces.Tuple((zero_piston_space, zero_tip_space, zero_tilt_space)) 

        zero_secondaries_tuple = tuple([zero_ptt_space] * self.optical_system.num_apertures)

        zero_secondaries_space = spaces.Tuple(zero_secondaries_tuple)

        zero_tensioner_space = spaces.Box(
            low=0.0,
            high=0.0,
            shape=(1,),
            dtype=np.float32
        )

        zero_tensioners_tuple = tuple([zero_tensioner_space] * self.optical_system.num_tensioners)

        zero_tensioners_space = spaces.Tuple(zero_tensioners_tuple)
        
        zero_single_command_space = spaces.Tuple(
            (zero_secondaries_space, zero_tensioners_space)
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


    def assign_value_by_tree_address(self, action_space_list, value, tree_address):

        # Define the string of indices
        indices_str = "0_1_0_1"

        # Convert the string to a list of integers
        indices = list(map(int, tree_address.split('_')))

        # Use the indices to assign a new value to the corresponding element in the nested list
        sublist = action_space_list
        for index in indices[:-1]:
            sublist = sublist[index]
        sublist[indices[-1]] = value


        return action_space

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

            # assign_value_by_tree_address(action_space_list, action_value, tree_address)

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

        # Set the initial state. This is the first thing called in an episode.
        print("=== Start: Reset Environment ===")
        print("Instantiating a New Optical System")

        self.build_optical_system(**self.kwargs)

        # Calibrate the DM interaction matrix.
        self.optical_system.calibrate_dm_interaction_matrix()
        rcond = 1e-3
        self.reconstruction_matrix = hcipy.inverse_tikhonov(
            self.optical_system.interaction_matrix.transformation_matrix,
            rcond=rcond)

        self.episode_time_ms = 0.0
    
        print("Populating Initial Action")


        # self.action = np.zeros_like(self.action_space.sample())
        self.action = self.zero_action_space.sample()

        print("Populating Initial State")

        # TODO: Compute the calibration noise level to generate a sample.
        calibration_noise_nm = 10.0
        calibration_noise_microns = calibration_noise_nm / 1000
        calibration_noise_counts = self.microns_opd_per_actuator_bit / \
            calibration_noise_microns
        
        # Build a DM command that corresponds to the calibration noise.
        # TODO: refactor to remove sampling.
        # ones_like_action = np.ones_like(self.action_space.sample())
        # zeros_like_action = np.zeros_like(self.action_space.sample())
        # dm_calibration_noise = np.random.normal(
        #     loc=zeros_like_action,
        #     scale=calibration_noise_counts * ones_like_action
        # )
        # dm_calibration_noise_counts = dm_calibration_noise.astype(np.int16)


        if self.randomize_dm:
            
            print("Randomizing DM")
            self.optical_system.randomize_dm()

        for _ in range(self.frames_per_decision):

            (initial_state, _, _, _, info) = self.step(
                action=self.zero_action_space.sample(),
                noisy_command=False,
                ao_loop_active=False,
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

        deepcopy_shwfs_slopes = copy.deepcopy(self.shwfs_slopes)
        self.state_content["shwfs_slopes"].append(
            deepcopy_shwfs_slopes
        )

        if self.report_time:
            print("- Deepcopy time:   %.6f" %
                    (time.time() - deepcopy_start))

    def step(self,
             action,
             ao_loop_active=True,
             noisy_command=False,
             reset=False):


        if self.report_time:
            step_time = time.time()


        # First, ensure the step action is valid.
        assert self.action_space.contains(action), \
               "%r (%s) invalid"%(action, type(action))

        # Clear the custom state content for population.
        # TODO: encapsulate this mess...
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
        # structured_action = self.flat_to_dict(action, self.dict_action_space)
        self.action = self.flat_to_dict(action, self.dict_action_space)
        
        # TODO: Replace this with a static property.
        self.focal_plane_images = list()
        self.shwfs_slopes_list = list()

        # Run the step simulation loop.
        for frame_num in range(self.frames_per_decision):

            # print("-Running frame %s of %s." % (frame_num + 1,
            #                                     self.frames_per_decision))

            # Create a blank frame for manual integration.
            frame = np.zeros(self.image_shape, dtype=np.float64)

            # Iterate over each command, applying it and integrating the frame.
            for command_num in range(self.commands_per_frame):

                self.episode_time_ms += self.control_interval_ms
                self.state_content["action_times"].append(self.episode_time_ms)

                # Evolve the atmosphere to the current time.
                atmospere_evolution_start = time.time()
                self.optical_system.evolve_atmosphere_to(self.episode_time_ms)
                if self.report_time:
                    print("- Atmosphere time: %.6f" % (time.time() - atmospere_evolution_start))

                # Get the commanded actuations.
                # TODO: Add direct dm command support to enable focal plane ao.
                command = self.action[command_num]
                    
                # TODO: externalize this.
                command_tensioners = True
                if command_tensioners:

                    tensioner_commands = command[1]
                    self.optical_system.command_tensioners(tensioner_commands)

                # TODO: externalize this.
                command_secondaries = True
                if command_secondaries:

                    secondaries_commands = command[0]
                    self.optical_system.command_secondaries(secondaries_commands)

                # TODO: externalize this.
                command_dm = False
                if command_dm:

                    raise NotImplementedError("Agent DM command isn't implemented.")
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
                    if ao_loop_active:

                        # Measure the wavefront using the SHWFS.
                        shwfs_readout_vector = self.optical_system.get_shwfs_frame(
                            integration_seconds=integration_seconds
                        )
                        
                        # TODO: Record the slopes.
                        # Compute the correction slopes for the DM.
                        shwfs_slopes = self.optical_system.shwfse.estimate([shwfs_readout_vector + 1e-10])
                        shwfs_slopes -= self.optical_system.reference_slopes
                        self.shwfs_slopes = shwfs_slopes.ravel()
                        self.shwfs_slopes_list.append(self.shwfs_slopes)

                        # Perform wavefront compensation by setting the DM actuators.
                        self.optical_system.dm.actuators = (1 - self.dm_leakage) * self.optical_system.dm.actuators - self.dm_gain * self.reconstruction_matrix.dot(self.shwfs_slopes)

                    if self.report_time:
                        print("-- AO Step time: %.6f" % (time.time() - ao_step_time_start))

                    # Get a fractional science frame.
                    science_readout_vector = self.optical_system.get_science_frame(
                        integration_seconds=integration_seconds
                    )
                    self.science_readout_raster = np.reshape(science_readout_vector,
                                                             self.image_shape)

                    # Note: This step accumulates the partial readout rasters, in effect manually
                    #       integrating them outside of HCIPy.
                    frame += self.science_readout_raster

                    # Deepcopy the environment state so that it can be stored later.
                    if self.record_env_state_info and not reset:
                        self.save_state()
                    
            # Finally, append this frame to the stack of focal plane images.
            self.focal_plane_images.append(frame)

        # Set the state to focal plane image.
        self.state = self.focal_plane_images

        # self.reward_function = "ao_rms_slope"

        if self.reward_function == "strehl":

            # raise NotImplementedError("The Strehl reward isn't implemented.")
            # Marechal approximation.

            strehls = list()
        
            for focal_plane_image in self.focal_plane_images:
                # print("strehl")
                # print(hcipy.metrics.get_strehl_from_focal(
                #     focal_plane_image.flatten() / np.max(focal_plane_image),
                #     self.optical_system.perfect_image / np.max(self.optical_system.perfect_image)
                # ))

                # self.optical_system.perfect_image
                # # plt.ion()
                # fig = plt.figure()
                # plt.subplot(2, 1, 1)
                # plt.title('focal_plane_image')
                # plt.imshow(focal_plane_image, cmap='inferno') #
                # plt.colorbar()
            
                # plt.subplot(2, 1, 2)
                # plt.title('perfect_image')
                # plt.imshow(np.reshape(self.optical_system.perfect_image, self.image_shape), cmap='inferno')
                # plt.colorbar()
                # plt.show()
                # die
            
                strehls.append(hcipy.metrics.get_strehl_from_focal(
                    focal_plane_image.flatten() / np.max(focal_plane_image),
                    self.optical_system.perfect_image / np.max(self.optical_system.perfect_image)
                ))

            reward = np.mean(strehls)

        elif self.reward_function == "ao_rms_slope":

            reward = 0.0

            if ao_loop_active:
                
                for shwfs_slopes in self.shwfs_slopes_list:

                    # TODO: replace the numerator with the max inverse rms slope.

                    reward += 1 / np.sqrt(np.mean(shwfs_slopes ** 2))

            else:

                reward = 0.0

        elif self.reward_function == "norm_ao_rms_slope":

            reward = 0.0

            if ao_loop_active:
                
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

            if ao_loop_active:
                
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

