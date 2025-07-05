import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Binomial


import os

import tyro
from dataclasses import dataclass





@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 88
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if using cuda, set the gpu"""
    gpu_list: int = 0
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    num_envs: int = 1
    """The number of environments to create."""
    async_env: bool = False
    """Whether to use an AsynchronousVectorEnv"""
    subproc_env: bool = False
    """Whether to use a SubprocVectorEnv"""
    model_save_interval: int = 100
    """The interval between saving model weights"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    # learning_rate: float = 3e-4
    actor_learning_rate: float = 1e-5
    """the learning rate of the optimizer"""
    critic_learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e5)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.004
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 16
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 256
    """timestep to start learning"""
    policy_frequency: int = 1
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    decay_rate: float = 0.00001
    """Decay rate for noise decay"""

    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""

    # Actor model parameters
    actor_type: str = "vanilla"
    """Type of actor model to use."""
    actor_channel_scale: int = 16
    """The scale of the actor model channels."""
    actor_fc_scale: int = 64
    """The scale of the actor model fully connected layers."""
    low_dim_actor: bool = False
    """Whether the actor model is visual."""

    # QNetwork model parameters
    critic_type: str = "vanilla"
    """Type of QNetwork model to use."""
    qnetwork_channel_scale: int = 16
    """The scale of the QNetwork model channels."""
    qnetwork_fc_scale: int = 64
    """The scale of the QNetwork model fully connected layers."""
    low_dim_qnetwork: bool = False
    """Whether the qnetwork model is visual."""


    # Custom Algorthim Arguments
    """Which prelearning sample strategy to use (e.g., 'scales', 'normal')"""
    prelearning_sample: str = ""

    # visual pendulum parameters
    # learning_rate: float = 3e-4
    # """the learning rate of the optimizer"""
    # buffer_size: int = int(1e6)
    # """the replay memory buffer size"""
    # gamma: float = 0.99
    # """the discount factor gamma"""
    # tau: float = 0.004
    # """target smoothing coefficient (default: 0.005)"""
    # batch_size: int = 64
    # """the batch size of sample from the reply memory"""
    # exploration_noise: float = 0.1
    # """the scale of exploration noise"""
    # learning_starts: int = 256
    # """timestep to start learning"""
    # policy_frequency: int = 4
    # """the frequency of training policy (delayed)"""
    # noise_clip: float = 0.5
    # """noise clip parameter of the Target Policy Smoothing Regularization"""

    # Environment specific arguments
    """Class for holding all arguments for the script."""
    gpu_list: str = "0"
    """The list of GPUs to use."""
    render: bool = False
    """Whether to render the environment."""
    report_time: bool = False
    """Whether to report time statistics."""
    action_type: str = "none"
    """The type of action to use."""
    object_type: str = "binary"
    """The type of object to use."""
    aperture_type: str = "elf"
    """The type of aperture to use."""
    max_episode_steps: int = 100


    """Toggle to enable discrete control."""
    discrete_control: bool = False
    """The number of discrete control steps."""
    discrete_control_steps: int = 128
    """Toggle to enable incremental control."""
    incremental_control: bool = False
    """Toggle to enable agent control of tensioners."""
    command_tensioners: bool = False
    """Toggle to enable agent control of secondaries."""
    command_secondaries: bool = False
    """Toggle to enable agent control of tip/tilt for large mirrors."""
    command_tip_tilt: bool = False
    """Toggle to enable agent control of dm."""
    command_dm: bool = False

    init_differential_motion: bool = False
    """Whether to initialize differential motion."""
    simulate_differential_motion: bool = False
    """Whether to simulate differential motion."""


    """ The type of observation to model 'image_only' or 'image_action'."""
    observation_mode: bool = "image_only"

    """The type of aperture to use."""
    ao_loop_active: bool = False
    """The maximum number of steps per episode."""
    num_episodes: int = 1
    """The number of episodes to run."""
    num_atmosphere_layers: int = 0
    """The number of atmosphere layers."""
    reward_threshold: float = 25.0
    """The reward threshold to reach."""
    num_steps: int = 16
    """The number of steps to take."""
    silence: bool = False
    """Whether to silence the output."""
    optomech_version: str = "test"
    """The version of optomech to use."""
    reward_function: str = "strehl"
    """The reward function to use."""
    render_frequency: int = 1
    """The frequency of rendering."""
    ao_interval_ms: float = 1.0
    """The interval between AO updates."""
    control_interval_ms: float = 1.0
    """The interval between control updates."""
    frame_interval_ms: float = 1.0
    """The interval between frames."""
    decision_interval_ms: float = 1.0
    """The interval between decisions."""
    focal_plane_image_size_pixels: int = 256
    """The size of the focal plane image in pixels."""
    render_dpi: float = 500.0
    """The DPI for rendering."""
    record_env_state_info: bool = False
    """Whether to record environment state information."""
    write_env_state_info: bool = False
    """Whether to write environment state information."""
    state_info_save_dir: str = "./tmp/"
    """The directory to save state information."""
    randomize_dm: bool = False
    """Whether to randomize the DM."""
    extended_object_image_file: str = ".\\resources\\sample_image.png"
    """The file for the extended object image."""
    extended_object_distance: str = None
    """The distance to the extended object."""
    extended_object_extent: str = None
    """The extent of the extended object."""
    observation_window_size: int = 2**1
    """The size of the observation window."""
    num_tensioners: int = 16
    """The number of tensioners."""
    model_wind_diff_motion: bool = False
    """Whether to model wind differential motion."""
    model_gravity_diff_motion: bool = False
    """Whether to model gravity differential motion."""
    model_temp_diff_motion: bool = False
    """Whether to model temperature differential motion."""


#PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np

#Hyperparameters
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 2
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, observation_space, action_space):
        super(PPO, self).__init__()
        self.data = []
        obs_image_shape = observation_space.shape

        # Check if this is a pixels environment
        if len(obs_image_shape) >= 3:
            self.pixels = True
        else:
            self.pixels = False

        if self.pixels:
            # Check if this is a channels-last environment
            if obs_image_shape[-1] < obs_image_shape[0]:
                self.channels_last = True
                input_channels = obs_image_shape[-1]
            else:
                self.channels_last = False
                input_channels = obs_image_shape[0]

            self.visual_encoder = nn.Sequential(
                nn.Conv2d(input_channels, 
                        16,
                        kernel_size=7,
                        stride=2),
                nn.ReLU(),
                nn.Conv2d(16,
                        32,
                        kernel_size=5,
                        stride=1),
                nn.ReLU(),
                nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        stride=1),
                nn.ReLU(),
                # nn.Conv2d(64,
                #         64,
                #         kernel_size=5,
                #         stride=2),
                # nn.ReLU(),
                # nn.Flatten(),
            )


            # Get the output shape of the visual encoder
            with torch.inference_mode():

                # Handle channels-last environments.
                x = torch.zeros(1, *obs_image_shape)
                if self.channels_last:
                    x = x.permute(0, 3, 1, 2)
                visual_output_shape = self.visual_encoder(x).shape
                print("Visual output shape:", visual_output_shape)
        # die
        if self.pixels:
            self.fc1   = nn.Linear(int(np.prod(visual_output_shape)), 64)
        else:
            self.fc1   = nn.Linear(4, 64)
        self.lstm  = nn.LSTM(64,32)



        if self.pixels:
            self.action_size = action_space.shape[0]
        else:

             self.action_size = 2

        self.fc_pi = nn.Linear(32,  self.action_size)
        self.fc_v  = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        if self.pixels:
            x = self.visual_encoder(x)

        print("x", int(np.prod(x.shape)))
        # die
        x = F.relu(self.fc1(torch.flatten(x)))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden
    
    def v(self, x, hidden):
        if self.pixels:
            x = self.visual_encoder(x)
            
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]
        
    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()
            
            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()
        
def main(args):
    env_id = 'optomech-v1'
    if env_id == 'CartPole-v1':
        env = gym.make('CartPole-v1',)
    elif env_id == 'optomech-v1':
        env = gym.make('optomech-v1', **vars(args))
    



    # Check if MPS is available
    if torch.cuda.is_available():
        print("Running with CUDA")
        device = torch.device("cuda")
        torch.cuda.set_device(int(args.gpu_list))
    elif torch.backends.mps.is_available():
        print("Running with MSP")
        device = torch.device("mps")
    else:
        print("Running with CPU")
        device = torch.device("cpu")

    # TODO: test with GPU and if its fine, don't worry about it.
    device = torch.device("cpu")

    model = PPO(observation_space=env.observation_space,
                action_space=env.action_space).to(device)
    score = 0.0
    print_interval = 20
    
    for n_epi in range(10000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        s = env.reset()
        done = False
        
        while not done:
            for t in range(T_horizon):
                h_in = h_out



                # prob, h_out = model.pi(torch.from_numpy(s[0]).float().to(device), h_in)
                if env_id == 'CartPole-v1':
                    prob, h_out = model.pi(torch.from_numpy(s).float().to(device), h_in)
                elif env_id == 'optomech-v1':
                    prob, h_out = model.pi(torch.from_numpy(s[0]).float().to(device), h_in)
                prob = prob.view(-1)
                print(prob)
                #[torch.multinomial(torch.softmax(l, dim=1), num_samples=1) for l in prob]

                m = Binomial(1, prob)
                # print(m)
                a = m.sample().int()
                print(a)
                print(env.action_space)
                s_prime, r, done, info = env.step(np.array(a))

                model.put_data((s, a, r/100.0, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime

                score += r
                if done:
                    break
                    
            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            score = 0.0

    env.close()

if __name__ == '__main__':

    args = tyro.cli(Args)

    # Register our custom optomech environment.
    gym.envs.registration.register(
        id='optomech-v1',
        entry_point='optomech.optomech:OptomechEnv',
    )

    main(args)