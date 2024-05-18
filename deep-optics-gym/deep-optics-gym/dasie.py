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
    gaussian1D = signal.gaussian(n, std)
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
        kernel_extent = 8 * std

        # TODO: make mu_y and mu_x determined by source_position.
        # mu_x = self.extent_pixels // 4
        # mu_y = self.extent_pixels // 4
        mu_x = self.extent_pixels // 2
        mu_y = self.extent_pixels // 2

        return offset_gaussian(self.extent_pixels,
                               mu_x,
                               mu_y,
                               std,
                               kernel_extent,
                               normalised=True)

    def rgb2gray(self, rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


class OpticalSystem(object):

    def __init__(self, **kwargs):

        self.report_time = kwargs["report_time"]
        self.microns_opd_per_actuator_bit = 0.00015

        # Parameters for the pupil function
        focal_length = 200.0 # m
        pupil_diameter = 3.6 # m
        elf_segment_centroid_diameter = 2.5 # m
        
        # Parameters for the optical simulation.
        num_pupil_grid_simulation_pixels = kwargs['focal_plane_image_size_pixels']
        self.wavelength = 763e-9
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

        fried_parameter = hcipy.seeing_to_fried_parameter(seeing)
        print("fried_parameter: %s" % fried_parameter)
        Cn_squared = hcipy.Cn_squared_from_fried_parameter(
            fried_parameter,
            wavelength=self.wavelength
        )
        velocity = 0.314 * fried_parameter / tau0

        # Extract the provided focal plane pararmeters.
        num_focal_grid_pixels = kwargs['focal_plane_image_size_pixels']
        # TODO: Externalize.
        kwargs['focal_plane_image_size_meters'] = 8.192  * 1e-3
        focal_plane_extent_metres = kwargs['focal_plane_image_size_meters']
        
        focal_plane_resolution_element = self.wavelength * focal_length / pupil_diameter
        focal_plane_pixels_per_meter = num_focal_grid_pixels / focal_plane_extent_metres
        focal_plane_pixel_extent_meters = focal_plane_extent_metres /  num_focal_grid_pixels
        self.ifov = 206265 / focal_length * focal_plane_pixel_extent_meters


        # This combination sets the focal 
        # sampling: The number of pixels per resolution element (= lambda f / D).
        sampling = focal_plane_resolution_element / focal_plane_pixel_extent_meters
        # num_airy: The spatial extent of the grid in radius in resolution elements (= lambda f / D).
        num_airy = num_focal_grid_pixels / (sampling * 2)

        # sampling = 4
        # num_airy = num_focal_grid_pixels / (sampling * 2)
        # Print many computed values to aid in debugging.
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

        # Note: this variable toggles an experimental feature.
        use_geometric_optics = True
        if not use_geometric_optics:

            # First, instantiate all of the optical elements.
            self.object_grid = hcipy.make_uniform_grid(
                dims=[num_pupil_grid_simulation_pixels,
                      num_pupil_grid_simulation_pixels],
                extent=[self.object_plane.extent_meters]
            )

            object_field = self.make_object_field(self.object_plane.array)

            self.object_field = hcipy.evaluate_supersampled(object_field,
                                                            self.object_grid,
                                                            16)

            self.object_wavefront = hcipy.Wavefront(self.object_field,
                                                    self.wavelength)
            
            # Instantiate a Fresnel propagator from the object to pupil plane.
            self.object_to_pupil_propagator = hcipy.FresnelPropagator(
                self.object_grid,
                distance=self.object_plane.distance_meters
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
                num_apertures=15,
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

        else:

            # Note: if you add aperture types, update this exception.
            raise NotImplementedError(
                "aperture_type was %s, but only 'elf' and 'circular' are \
                implemented." % aperture_type)
        
        # TODO: Major upgrade: we must add the secondaries for ELF. Thus, this will
        #       need to become aperture-dependent. It may be best to refactor from 
        #       'aperture' to 'telescope' once these become coupled for clarity.

        # TODO: Major upgrade: we must add a SHWFS here.
        f_number = 50
        num_lenslets = 40 # 40 lenslets along one diameter
        sh_diameter = 5e-3 # m

        magnification = sh_diameter / pupil_diameter
        magnifier = hcipy.Magnifier(magnification)
        shwfs = hcipy.SquareShackHartmannWavefrontSensorOptics(
            self.pupil_grid.scaled(magnification),
            f_number,
            num_lenslets,
            sh_diameter)
        shwfse = hcipy.ShackHartmannWavefrontSensorEstimator(shwfs.mla_grid,
                                                             shwfs.micro_lens_array.mla_index)
        self.shwfs_camera = hcipy.NoiselessDetector(focal_grid)

        # Build a DM on the pupil grid.
        self.dm_influence_functions = hcipy.make_gaussian_influence_functions(
            self.pupil_grid,
            num_actuators_across_pupil=35,
            actuator_spacing=pupil_diameter / 35
        )
        self.dm = hcipy.DeformableMirror(self.dm_influence_functions)
        
        # Finally, make a camera.
        # Note: The camera is noiseless here because we can add noise in the Env step().
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
        self.dm.actuators = meters_opd_per_actuator_bit * command_vector

        return
    
    def get_frame(self, integration_seconds=1.0):

        

        # Chain together the wavefronts using the optical elements to produce a frame.

        # Make a pupil plane wavefront from aperture
        # Note: this variable toggles an experiemntal feature.
        use_geometric_optics = True

        object_wavefront_start_time = time.time()
        # If I comment this, the focal plane images are garbage.
        if use_geometric_optics:

            self.object_wavefront = hcipy.Wavefront(self.aperture,
                                                    self.wavelength)
            self.pre_atmosphere_object_wavefront = self.object_wavefront

        else:

            self.pre_atmosphere_object_wavefront = self.object_to_pupil_propagator(
                self.object_wavefront
            )

        
        if self.report_time:
            print("-- Object Wavefront time: %0.6f" % (time.time() - object_wavefront_start_time))

        atmosphere_forward_start_time = time.time()
        # Propagate the object plane wavefront through the atmosphere layers.
        wf = self.pre_atmosphere_object_wavefront
        for atmosphere_layer in self.atmosphere_layers:
            wf = atmosphere_layer.forward(wf)
        self.post_atmosphere_wavefront = wf

        if self.report_time:
            print("-- Atmosphere Forward time: %0.6f" % (time.time() - atmosphere_forward_start_time))


        segmented_mirror_forward_start_time = time.time()
        # Apply the segmented mirror pupil to the post-atmosphere wavefront.
        self.pupil_wavefront = self.segmented_mirror(
            self.post_atmosphere_wavefront
        )

        if self.report_time:
            print("-- Segments Forward time: %0.6f" % (time.time() - segmented_mirror_forward_start_time))



        dm_forawrd_start_time = time.time()
        # Propagate the wavefront from the segmented mirror through the DM.
        # Note: counter-intuitively, the DM must be re-applied after changes.
        self.post_dm_wavefront = self.dm.forward(self.pupil_wavefront)

        if self.report_time:
            print("-- DM Forward time: %0.6f" % (time.time() - dm_forawrd_start_time))
        # self.post_dm_wavefront = self.dm(self.post_atmosphere_wavefront)

        # TODO: Add Fresnel prop m1 -> m2 (focal length: tbd)
        # Propagate from the pupil (M1) to the DM (M2).
      
        pupil_focal_prop_start_time = time.time()
        # Propagate from the DM (M2) to the focal (image) plane.
        # Note: counter-intutively, the propagator must be re-applied as well.
        # TODO: rename pupil_to_focal_propagator to dm_to_focal_propagator
        self.focal_plane_wavefront = self.pupil_to_focal_propagator(
            self.post_dm_wavefront
        )

        if self.report_time:
            print("-- Pupil-Focal Propagation time: %0.6f" % (time.time() - pupil_focal_prop_start_time))


        integration_start_time = time.time()
        # Integrate the wavefront, read out, and return. 
        self.camera.integrate(self.focal_plane_wavefront, integration_seconds)

        if self.report_time:
            print("-- Camera Integration time: %0.6f" % (time.time() - integration_start_time))


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
        self.render_mode = kwargs['render_mode']
        self.render_dpi = kwargs['render_dpi']
        self.record_env_state_info = kwargs['record_env_state_info']

        # Parse simulation parameters.
        self.render_frequency = kwargs['render_frequency']
        self.control_interval_ms = kwargs['control_interval_ms']
        self.frame_interval_ms = kwargs['frame_interval_ms']
        self.decision_interval_ms = kwargs['decision_interval_ms']
        self.ao_interval_ms = kwargs['ao_interval_ms']
        # TODO: Externalize.
        self.microns_opd_per_actuator_bit = 0.00015
        self.stroke_count_limit = 20000

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

        # Print the provided intervals
        print("Control interval: %s ms" % self.control_interval_ms)
        print("Frame interval: %s ms" % self.frame_interval_ms)
        print("Decision interval: %s ms" % self.decision_interval_ms)
        print("AO interval: %s ms" % self.ao_interval_ms)

        # Compute the number of commands per decision and frame, rounding up.
        self.commands_per_decision = math.ceil(
            self.decision_interval_ms / self.control_interval_ms
        )
        self.commands_per_frame = math.ceil(
            self.frame_interval_ms / self.control_interval_ms
        )
        self.frames_per_decision = math.ceil(
            self.decision_interval_ms / self.frame_interval_ms
        )

        print("Commands per decision: %s" % self.commands_per_decision)
        print("Commands per frame: %s" % self.commands_per_frame)
        print("Frames per decision: %s" % self.frames_per_decision)

        # Create a dict to hold the some hidden state content.
        self.build_optical_system(**kwargs)
        self.episode_time_ms = 0.0

        # Build the command grid, which is one-to-one with the action space.
        # TODO: In the active optics formulation, this needs to be ttp secondaries and tensioners.
        # TODO: Retain the ability to control any subset fo actuators.
        self.actuator_command_grid = np.zeros(
            shape=(35, 35),
            dtype=np.int16
        )

        # Define a symmetric action space.
        action_shape = (self.commands_per_decision,
                        self.actuator_command_grid.shape[0],
                        self.actuator_command_grid.shape[1],)
        self.action_space = spaces.Box(low=-self.stroke_count_limit,
                                       high=self.stroke_count_limit,
                                       shape=action_shape,
                                       dtype=np.int16)

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

        print("==== End: Initializing Environment ====")

    def seed(self, seed=None):

        self.np_random, seed = seeding.np_random(seed)

        return [seed]

    def reset(self):

        # Set the initial state. This is the first thing called in an episode.
        print("=== Start: Reset Environment ===")
        print("Instantiating a New Optical System")

        self.build_optical_system(**self.kwargs)

        self.episode_time_ms = 0.0
    
        print("Populating Initial Action")

        self.action = np.zeros_like(self.action_space.sample())

        print("Populating Initial State")

        # TODO: Compute the calibration noise level to generate a sample.
        calibration_noise_nm = 10.0
        calibration_noise_microns = calibration_noise_nm / 1000
        calibration_noise_counts = self.microns_opd_per_actuator_bit / \
            calibration_noise_microns
        
        # Build a DM command that corresponds to the calibration noise.
        # TODO: refactor to remove sampling.
        ones_like_action = np.ones_like(self.action_space.sample())
        zeros_like_action = np.zeros_like(self.action_space.sample())
        dm_calibration_noise = np.random.normal(
            loc=zeros_like_action,
            scale=calibration_noise_counts * ones_like_action
        )
        dm_calibration_noise_counts = dm_calibration_noise.astype(np.int16)
        for _ in range(self.frames_per_decision):

            (initial_state, _, _, _, _) = self.step(
                action=dm_calibration_noise_counts,
                noisy_command=False
            )

        self.state = initial_state
        self.steps_beyond_done = None

        print("=== End: Reset Environment ===")
        return np.array(self.state)


    def step(self,
             action,
             noisy_command=True,):

        if self.report_time:
            step_time = time.time()

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


        # Update the current action to be the provided action.
        self.action = action 
        
        # First, ensure the step action is valid.
        assert self.action_space.contains(action), \
               "%r (%s) invalid"%(action, type(action))

        # TODO: Replace this with a static property.
        self.focal_plane_images = list()

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
                dm_command = action[command_num]
                
                # If this command is noisy. Otherwise, don't.
                if noisy_command:

                    # Add noise to the command.
                    # TODO: Externalize this.
                    ninety_five_percentile_noise = 8
                    dm_command_noise_count_std = ninety_five_percentile_noise / 2 
                    dm_command_noise = np.random.normal(
                        loc=np.zeros_like(dm_command),
                        scale=dm_command_noise_count_std * np.ones_like(dm_command)
                    )
                    dm_command_noise_counts = dm_command_noise.astype(np.int16)
                    noisy_dm_command = dm_command + dm_command_noise_counts
                
                    # Apply the command to the DM.
                    self.optical_system.command_dm(noisy_dm_command)

                else:
                    
                    # Apply the command to the DM.
                    self.optical_system.command_dm(dm_command)

                # TODO: Major Feature. Add corrections_per_command inner loop and apply SHWFS
                #       AO system corrections for that number of iterations. This will require
                #       moving the partial frame integration step, below, into that loop. To 
                #       test this feature, the AO system should close without intervention when
                #       the Fried parameter and induced aberation is sufficiently small, but fail
                #       as it increases.

                # Manually integrate the frame to model dynamic optics.
                frame_interval_seconds = (self.frame_interval_ms / 1000.0)
                integration_seconds = frame_interval_seconds / self.commands_per_frame
                
                if self.report_time:
                    simulation_time_start = time.time()

                readout_vector = self.optical_system.get_frame(
                    integration_seconds=integration_seconds
                )

                if self.report_time:
                    print("- Simulation time: %.6f" % (time.time() - simulation_time_start))
                
                readout_raster = np.reshape(readout_vector, self.image_shape)

                # Note: This step accumulates the partial readout rasters, in effect manually
                #       integrating them outside fo HCIPy.
                frame += readout_raster
                
                if self.record_env_state_info:
                    # TODO: Encapsulate this mess...
                    # We have now completed the substep; store the state variables.
                    deepcopy_start = time.time()
                    self.state_content["dm_surfaces"].append(
                        copy.deepcopy(self.optical_system.dm.surface)
                    )

                    # TODO: Add support for saving the rest of the atmosphere layers.
                    if len(self.optical_system.atmosphere_layers) > 0:
                        self.state_content["atmos_layer_0_list"].append(
                            copy.deepcopy(self.optical_system.atmosphere_layers[0])
                        )
                    else:
                        self.state_content["atmos_layer_0_list"].append(None)

                    self.state_content["object_fields"].append(
                        copy.deepcopy(self.optical_system.object_plane)
                    )

                    self.state_content["pre_atmosphere_object_wavefronts"].append(
                        copy.deepcopy(self.optical_system.pre_atmosphere_object_wavefront)
                    )
                    
                    self.state_content["post_atmosphere_wavefronts"].append(
                        copy.deepcopy(self.optical_system.post_atmosphere_wavefront)
                    )
                    
                    self.state_content["segmented_mirror_surfaces"].append(
                        copy.deepcopy(self.optical_system.segmented_mirror.surface)
                    )
                    
                    self.state_content["pupil_wavefronts"].append(
                        copy.deepcopy(self.optical_system.pupil_wavefront)
                    )
                    
                    self.state_content["post_dm_wavefronts"].append(
                        copy.deepcopy(self.optical_system.post_dm_wavefront)
                    )
                    
                    self.state_content["focal_plane_wavefronts"].append(
                        copy.deepcopy(self.optical_system.focal_plane_wavefront)
                    )
                    
                    self.state_content["readout_images"].append(
                        copy.deepcopy(readout_raster)
                    )

                    if self.report_time:
                        print("- Deepcopy time:   %.6f" %
                                (time.time() - deepcopy_start))

            # Finally, append this frame to the stack of focal plane images.
            self.focal_plane_images.append(frame)

        # Set the state to focal plane image.
        self.state = self.focal_plane_images

        # TODO: compute_reward()
        # TODO: Major Feature. Add a closed-loop SHWFS AO system and use the fact of
        #       its closure as the reward.
        self.reward_function = "unity"

        if self.reward_function == "strehl":

            raise NotImplementedError("The Strehl reward isn't implemented.")

        elif self.reward_function == "unity":

            reward = 1.0

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

        return np.array(self.state), reward, terminated, truncated, info

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


    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def build_optical_system(self, **kwargs):

        self.optical_system = OpticalSystem(**kwargs)

        return

