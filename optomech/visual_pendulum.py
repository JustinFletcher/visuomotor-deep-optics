import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from PIL import Image
import matplotlib.pyplot as plt
import uuid

class VisualPendulumEnv(PendulumEnv):
    def __init__(self, resolution=16, render_style='observation', rod_width=None):
        """
        Initialize the VisualPendulumEnv.

        Parameters:
            resolution (int): The resolution of the generated image (default: 16).
            render_style (str): The rendering style for the environment.
                                Options are 'observation' for native observation visualization,
                                'pendulum' for visual rendering via gymnasium's renderer,
                                and 'fast_pendulum' for a fast numpy-only white-on-black render.
            rod_width (int or None): Width of the pendulum rod in pixels for 'fast_pendulum'.
                                     Defaults to max(1, resolution // 16).
        """
        # Only need rgb_array render mode for the slow 'pendulum' style
        super().__init__(render_mode='rgb_array')

        self.uuid = uuid.uuid4()

        self.resolution = resolution
        self.render_style = render_style

        # Observation space for grayscale image
        self.observation_space = Box(low=0, high=255, shape=(resolution, resolution, 1), dtype=np.uint8)

        # Define ranges for scaling the observation values in 'observation' style
        self.obs_ranges = {
            0: (-1.0, 1.0),   # Cosine of the angle
            1: (-1.0, 1.0),   # Sine of the angle
            2: (-8.0, 8.0),   # Angular velocity
        }

        # Pre-compute constants for fast_pendulum rendering
        if rod_width is None:
            rod_width = max(1, resolution // 16)
        self._rod_width = rod_width
        self._rod_length = 0.4  # fraction of resolution (pivot to tip)
        self._pivot = np.array([resolution / 2.0, resolution / 2.0])
        # Pre-allocate the output image buffer to avoid per-step allocation
        self._fast_img_buf = np.zeros((resolution, resolution, 1), dtype=np.uint8)

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.last_observation = observation  # Store the latest observation
        return self._get_visual_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        # print("raw outputs")
        # print(terminated)
        # print(truncated)
        self.last_observation = observation  # Update the latest observation
        return self._get_visual_observation(observation), reward, terminated, truncated, info

    def _scale_observation(self, value, obs_index):
        """Scales an observation value to the range [0, 255] based on its predefined range."""
        min_val, max_val = self.obs_ranges[obs_index]
        scaled_value = (value - min_val) / (max_val - min_val) * 255
        return np.clip(scaled_value, 0, 255)

    def _create_observation_image(self, observation):
        """
        Creates an image representation of the 3 native observation values, scaled to [0, 255].
        Each observation value occupies approximately one-third of the image width.
        """
        # Scale each observation to [0, 255]
        scaled_obs = [self._scale_observation(observation[i], i) for i in range(3)]
        
        # Create a blank grayscale image
        img = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        
        # Define the width of each observation section
        section_width = self.resolution // 3
        
        # Tile each scaled observation across its section of the image
        for i in range(3):
            img[:, i * section_width:(i + 1) * section_width] = scaled_obs[i]
            if i == 2:
                img[:, i * section_width:(i + 1) * section_width] = 0.0
        
        # Convert the image to 3D format (H, W, C) with a single channel
        return img[:, :, np.newaxis]

    def _render_fast_pendulum(self):
        """
        Render the pendulum as a white rod on a black background using pure
        numpy operations.  No PIL, no matplotlib, no super().render().

        The rod extends from the pivot (image centre) in the direction given
        by theta (self.state[0]).  Convention: theta=0 is pointing UP; the
        rod tip at angle theta is at:
            dx =  sin(theta)   (positive = right)
            dy = -cos(theta)   (positive = up, but image y is inverted)
        """
        img = self._fast_img_buf
        img[:] = 0  # clear to black

        theta = self.state[0]
        res = self.resolution
        rod_px = self._rod_length * res  # rod length in pixels

        # Tip position in image coordinates (y increases downward).
        # theta=0 means upright, so the rod points UP (negative y in image).
        # dx =  sin(theta), dy = -cos(theta) in math coords
        # In image coords (y flipped): dy_img = cos(theta) ... WAIT
        # Actually: math-y up => image-y down.  At theta=0 (upright),
        # the bob is ABOVE the pivot, so tip_y < cy  =>  tip_y = cy - rod_px.
        cx, cy = self._pivot
        tip_x = cx + rod_px * np.sin(theta)
        tip_y = cy - rod_px * np.cos(theta)  # negative cos => up in image at theta=0

        # Generate points along the rod using linear interpolation.
        # Use enough samples to cover every pixel along the rod.
        n_samples = int(rod_px * 2) + 2
        ts = np.linspace(0.0, 1.0, n_samples)
        xs = (cx + ts * (tip_x - cx)).astype(np.intp)
        ys = (cy + ts * (tip_y - cy)).astype(np.intp)

        # Widen the rod by offsetting perpendicular to the rod direction.
        # Perpendicular direction: (-dy, dx) normalised (already unit length
        # since (sin, cos) is unit).
        half_w = self._rod_width / 2.0
        if half_w >= 1.0:
            # Rod direction in image coords: (sin(theta), -cos(theta))
            # Perpendicular: (cos(theta), sin(theta))
            perp_x = np.cos(theta)
            perp_y = np.sin(theta)
            offsets = np.arange(-int(half_w), int(half_w) + 1)
            # Broadcast: xs[n_samples] + offsets[w] -> [w, n_samples]
            all_xs = xs[np.newaxis, :] + (offsets[:, np.newaxis] * perp_x).astype(np.intp)
            all_ys = ys[np.newaxis, :] + (offsets[:, np.newaxis] * perp_y).astype(np.intp)
            all_xs = all_xs.ravel()
            all_ys = all_ys.ravel()
        else:
            all_xs = xs
            all_ys = ys

        # Clip to image bounds and draw
        mask = (all_xs >= 0) & (all_xs < res) & (all_ys >= 0) & (all_ys < res)
        img[all_ys[mask], all_xs[mask], 0] = 255

        # Draw a small bright circle at the pivot (3x3 cross)
        pc = int(cx)
        pr = int(cy)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                py, px = pr + dy, pc + dx
                if 0 <= py < res and 0 <= px < res:
                    img[py, px, 0] = 180

        return img

    def _get_visual_observation(self, observation):
        """
        Returns the appropriate visual observation based on the selected render style.
        """
        if self.render_style == 'observation':
            return self._create_observation_image(observation)
        elif self.render_style == 'pendulum':
            return self._render_pendulum()
        elif self.render_style == 'fast_pendulum':
            return self._render_fast_pendulum()
        else:
            raise ValueError(
                f"Invalid render_style '{self.render_style}'. "
                "Use 'observation', 'pendulum', or 'fast_pendulum'."
            )

    def _render_pendulum(self):
        """
        Creates an image representation of the pendulum based on the rendered image.
        Resizes the rendered image to the specified resolution.
        """
        rendered_image = super().render()  # No mode argument
        if rendered_image is None:
            raise ValueError("Render mode is not producing an image. Ensure render_mode='rgb_array' is set.")
        
        # Resize the rendered RGB image to match the specified resolution
        img = Image.fromarray(rendered_image).resize((self.resolution, self.resolution))
        
        # Convert to grayscale and then expand dimensions to make it a single-channel image
        img = np.array(img.convert("L"))[:, :, np.newaxis]
        img = img.astype(np.uint8)  # Ensure the image is in uint8 format
        # Plot a historgram of the pixel values
        # plt.hist(img.flatten(), color='gray', alpha=0.7)
        # plt.title('Pixel Value Histogram')
        # plt.xlabel('Pixel Value')
        # plt.ylabel('Frequency')
        # plt.show()

        return img

    def render(self):
        """Renders the visual observation as an RGB image for compatibility with gym render modes."""
        observation = self._get_visual_observation(self.last_observation)
        return np.repeat(observation, 1, axis=2)  # Convert to RGB by repeating the channel


def main(render_style='pendulum'):
    """
    Run the VisualPendulumEnv environment with the specified render style.
    
    Parameters:
        render_style (str): The rendering style for the environment. Options are 'observation' and 'pendulum'.
    """
    # Instantiate the environment
    env = VisualPendulumEnv(resolution=16, render_style=render_style)
    
    # Reset the environment
    observation, _ = env.reset()
    
    # Plot the initial observation
    plt.imshow(observation.squeeze(), cmap='gray')
    plt.title(f"Initial Observation ({render_style} style)")
    plt.axis('off')
    plt.show()
    
    # Run the environment for a few steps
    for _ in range(128):
        action = env.action_space.sample()  # Sample a random action
        observation, _, _, _, _ = env.step(action)
        
        # Display the current observation
        plt.imshow(observation.squeeze(), cmap='gray')
        plt.title(f"Observation ({render_style} style) - Action: {action}")
        plt.axis('off')
        plt.show()

    env.close()


if __name__ == "__main__":
    # Test both render styles
    # print("Running with 'observation' style:")
    main()

    # print("Running with 'pendulum' style:")
    # main(render_style='pendulum')