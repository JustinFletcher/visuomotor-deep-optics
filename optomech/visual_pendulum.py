import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from PIL import Image
import matplotlib.pyplot as plt
import uuid

class VisualPendulumEnv(PendulumEnv):
    def __init__(self, resolution=32, render_style='pendulum'):
        """
        Initialize the VisualPendulumEnv.
        
        Parameters:
            resolution (int): The resolution of the generated image (default: 64).
            render_style (str): The rendering style for the environment. 
                                Options are 'observation' for native observation visualization 
                                and 'pendulum' for visual rendering of the pendulum.
        """
        super().__init__(render_mode='rgb_array')  # Set render mode to 'rgb_array' for visual rendering

        self.uuid = uuid.uuid4()

        self.resolution = resolution
        self.render_style = render_style  # Choose between 'observation' and 'pendulum'
        
        # Observation space for grayscale image
        self.observation_space = Box(low=0, high=255, shape=(resolution, resolution, 1), dtype=np.uint8)

        # Expand the action space by one dimension to accommodate the visual observation

        # Define ranges for scaling the observation values in 'observation' style
        self.obs_ranges = {
            0: (-1.0, 1.0),   # Cosine of the angle
            1: (-1.0, 1.0),   # Sine of the angle
            2: (-8.0, 8.0),   # Angular velocity
        }

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.last_observation = observation  # Store the latest observation
        return self._get_visual_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
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
        section_width = self.resolution // 2
        
        # Tile each scaled observation across its section of the image
        for i in range(3):
            img[:, i * section_width:(i + 1) * section_width] = scaled_obs[i]
        
        # Convert the image to 3D format (H, W, C) with a single channel
        return img[:, :, np.newaxis]

    def _get_visual_observation(self, observation):
        """
        Returns the appropriate visual observation based on the selected render style.
        """
        if self.render_style == 'observation':
            return self._create_observation_image(observation)
        elif self.render_style == 'pendulum':
            return self._render_pendulum()
        else:
            raise ValueError(f"Invalid render_style '{self.render_style}'. Use 'observation' or 'pendulum'.")

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
    env = VisualPendulumEnv(resolution=32, render_style=render_style)
    
    # Reset the environment
    observation, _ = env.reset()
    
    # Plot the initial observation
    plt.imshow(observation.squeeze(), cmap='gray')
    plt.title(f"Initial Observation ({render_style} style)")
    plt.axis('off')
    plt.show()
    
    # Run the environment for a few steps
    for _ in range(20):
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