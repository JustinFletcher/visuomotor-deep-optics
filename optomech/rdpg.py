'''
Recurrent Deterministic Policy Gradient (DDPG with LSTM network)
Update with batch of episodes for each time, so requires each episode has the same length.
'''


import math
import random
from pathlib import Path

import gymnasium as gym
import numpy as np

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical
from collections import namedtuple


# from common.value_networks import *
# from common.policy_networks import *
# from common.utils import *

import matplotlib.pyplot as plt
from matplotlib import animation
# from IPython.display import display
import argparse
from gym import spaces


# Start Buffers
import math
import random
import numpy as np
import torch



class ReplayBufferLSTM2:
    """ 
    Replay buffer for agent with LSTM network additionally storing previous action, 
    initial input hidden state and output hidden state of LSTM.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for LSTM initialization.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst=[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            ns_lst.append(next_state)
            d_lst.append(done)
            hi_lst.append(h_in)  # h_in: (1, batch_size=1, hidden_size)
            ci_lst.append(c_in)
            ho_lst.append(h_out)
            co_lst.append(c_out)
        hi_lst = torch.cat(hi_lst, dim=-2).detach() # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-2).detach()
        ci_lst = torch.cat(ci_lst, dim=-2).detach()
        co_lst = torch.cat(co_lst, dim=-2).detach()

        hidden_in = (hi_lst, ci_lst)
        hidden_out = (ho_lst, co_lst)

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)


    

# End Buffers

# Start Networks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import math

def linear_weights_init(m):
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)


def conv_weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class PolicyNetworkBase(nn.Module):
    """ Base network class for policy function """
    def __init__(self, state_space, action_space, action_range):
        super(PolicyNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  
        self._action_space = action_space
        self._action_shape = action_space.shape
        if len(self._action_shape) < 1:  # Discrete space
            self._action_dim = action_space.n
        else:
            self._action_dim = self._action_shape[0]
        self.action_range = action_range

    def forward(self):
        pass
    
    def evaluate(self):
        pass 
    
    def get_action(self):
        pass

    def sample_action(self,):
        a=torch.tensor(self._action_dim).uniform_(-1, 1)
        return self.action_range*a.numpy()
    
class DPG_PolicyNetworkLSTM2(PolicyNetworkBase):
    """
    Deterministic policy gradient network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, state_space, action_space, hidden_dim, action_range=1., init_w=3e-3):
        super().__init__(state_space, action_space, action_range)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, self._action_dim) # output dim = dim of action

        # weights initialization
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    

    def forward(self, state, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, action_dim)
        for lstm needs to be permuted as: (sequence_length, batch_size, -1)
        """
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        activation=F.relu
        # single branch
        x = torch.cat([state, last_action], -1)
        x = activation(self.linear1(x))   # lstm_branch: sequential data
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        x,  lstm_hidden = self.lstm1(x, hidden_in)    # no activation after lstm
        x = activation(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = x.permute(1,0,2)  # permute back

        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)

    def evaluate(self, state, last_action, hidden_in, noise_scale=0.0):
        '''
        evaluate action within GPU graph, for gradients flowing through it, noise_scale controllable
        '''
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = noise_scale * normal.sample(action.shape).to(device)
        action = self.action_range*action+noise
        return action, hidden_out

    def get_action(self, state, last_action, hidden_in,  noise_scale=1.0):
        '''
        select action for sampling, no gradients flow, noisy action, return .cpu
        '''
        state = torch.tensor(state).unsqueeze(0).unsqueeze(0).to(device) # increase 2 dims to match with training data
        last_action = torch.tensor(last_action).unsqueeze(0).unsqueeze(0).to(device)
        normal = Normal(0, 1)
        action, hidden_out = self.forward(state, last_action, hidden_in)
        noise = noise_scale * normal.sample(action.shape).to(device)
        action=self.action_range*action + noise
        return action.detach().cpu().numpy()[0][0], hidden_out

    def sample_action(self):
        normal = Normal(0, 1)
        random_action=self.action_range*normal.sample( (self._action_dim,) )

        return random_action.cpu().numpy()




class ValueNetworkBase(nn.Module):
    """ Base network class for value function approximation """
    def __init__(self, state_space, activation):
        super(ValueNetworkBase, self).__init__()
        self._state_space = state_space
        self._state_shape = state_space.shape
        if len(self._state_shape) == 1:
            self._state_dim = self._state_shape[0]
        else:  # high-dim state
            pass  

        self.activation = activation

    def forward(self):
        pass

class QNetworkBase(ValueNetworkBase):
    def __init__(self, state_space, action_space, activation ):
        super().__init__( state_space, activation)
        self._action_space = action_space
        self._action_shape = action_space.shape
        self._action_dim = self._action_shape[0]

class QNetworkLSTM2(QNetworkBase):
    """
    Q network with LSTM structure.
    The network follows single-branch structure as in paper: 
    Memory-based control with recurrent neural networks
    """
    def __init__(self, state_space, action_space, hidden_dim, activation=F.relu, output_activation=None):
        super().__init__(state_space, action_space, activation)
        self.hidden_dim = hidden_dim

        self.linear1 = nn.Linear(self._state_dim+2*self._action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # weights initialization
        self.linear3.apply(linear_weights_init)
        
    def forward(self, state, action, last_action, hidden_in):
        """ 
        state shape: (batch_size, sequence_length, state_dim)
        output shape: (batch_size, sequence_length, 1)
        for lstm needs to be permuted as: (sequence_length, batch_size, state_dim)
        """
        state = state.permute(1,0,2)
        action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        # single branch
        x = torch.cat([state, action, last_action], -1) 
        x = self.activation(self.linear1(x))
        x, lstm_hidden = self.lstm1(x, hidden_in)  # no activation after lstm
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        x = x.permute(1,0,2)  # back to same axes as input    
        return x, lstm_hidden    # lstm_hidden is actually tuple: (hidden, cell)   


# End Networks



# GPU = True
# device_idx = 0
# if GPU:
#     device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
# else:
#     device = torch.device("cpu")
# print(device)

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)

args = parser.parse_args()

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

print(device)

class RDPG():
    def __init__(self, replay_buffer, state_space, action_space, hidden_dim):
        self.replay_buffer = replay_buffer
        self.hidden_dim = hidden_dim
        # single-branch network structure as in 'Memory-based control with recurrent neural networks'
        self.qnet = QNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        self.target_qnet = QNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        self.policy_net = DPG_PolicyNetworkLSTM2(state_space, action_space, hidden_dim).to(device)
        self.target_policy_net = DPG_PolicyNetworkLSTM2(state_space, action_space, hidden_dim).to(device)

        # two-branch network structure as in 'Sim-to-Real Transfer of Robotic Control with Dynamics Randomization'
        # self.qnet = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.target_qnet = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.target_policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)

        print('Q network: ', self.qnet)
        print('Policy network: ', self.policy_net)

        for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)
        self.q_criterion = nn.MSELoss()
        q_lr=3e-4
        policy_lr = 1e-4
        self.update_cnt=0

        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    
    def target_soft_update(self, net, target_net, soft_tau):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        return target_net

    def update(self,
               batch_size,
               reward_scale=10.0,
               gamma=0.99,
               soft_tau=1e-3,
               policy_up_itr=10,
               target_update_delay=3,
               warmup=True):
        
        self.update_cnt += 1
        (hidden_in,
         hidden_out,
         state,
         action,
         last_action,
         reward,
         next_state,
         done) = self.replay_buffer.sample(batch_size)
        
        # INFO: I added this.
        tbtt_steps = 100  # time-steps for backprop through time

        # Time this step
        start = time.time()
        state      = np.array(state)[:, -tbtt_steps:, :]
        # print('state: ', state.shape)
        next_state = np.array(next_state)[:, -tbtt_steps:, :]
        action     = np.array(action)[:, -tbtt_steps:, :]
        last_action     = np.array(last_action)[:, -tbtt_steps:, :]
        reward     = np.array(reward)[:, -tbtt_steps:]
        done       = np.array(np.float32(done))[:, -tbtt_steps:]
        # Note: I added the np.arrays below. 
        # state      = torch.tensor(np.array(state)).to(device)
        # next_state = torch.tensor(np.array(next_state)).to(device)
        # action     = torch.tensor(np.array(action)).to(device)
        # last_action     = torch.tensor(np.array(last_action)).to(device)
        # reward     = torch.tensor(np.array(reward)).unsqueeze(-1).to(device)  
        # done       = torch.tensor(np.array(np.float32(done))).unsqueeze(-1).to(device)
        state      = torch.tensor(state).to(device)
        next_state = torch.tensor(next_state).to(device)
        action     = torch.tensor(action).to(device)
        last_action     = torch.tensor(last_action).to(device)
        reward     = torch.tensor(reward).unsqueeze(-1).to(device)  
        done       = torch.tensor(done).unsqueeze(-1).to(device)
        # print('Time to convert to tensors: ', time.time() - start)



        start = time.time()
        # use hidden states stored in the memory for initialization, hidden_in for current, hidden_out for target
        predict_q, _ = self.qnet(state, action, last_action, hidden_in) # for q 
        new_action, _ = self.policy_net.evaluate(state, last_action, hidden_in) # for policy
        new_next_action, _ = self.target_policy_net.evaluate(next_state, action, hidden_out)  # for q
        predict_target_q, _ = self.target_qnet(next_state, new_next_action, action, hidden_out)  # for q

        predict_new_q, _ = self.qnet(state, new_action, last_action, hidden_in) # for policy. as optimizers are separated, no detach for q_h_in is also fine
        target_q = reward+(1-done)*gamma*predict_target_q # for q
        # reward = reward_scale * (reward - reward.mean(dim=0)) /reward.std(dim=0) # normalize with batch mean and std
        # print('Time to predict: ', time.time() - start)

        start = time.time()
        q_loss = self.q_criterion(predict_q, target_q.detach())
        policy_loss = -torch.mean(predict_new_q)
        # print('Time to calculate loss: ', time.time() - start)

        # Note: I reversed the order here.
        # train policy_net
        start = time.time()
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()
        # print('Time to train policy: ', time.time() - start)

        # train qnet
        start = time.time()
        self.q_optimizer.zero_grad()
        q_loss.backward(retain_graph=True)  # no need for retain_graph here actually
        self.q_optimizer.step()
        # print('Time to train qnet: ', time.time() - start)
        

        # update the target_qnet
        start = time.time()
        if self.update_cnt%target_update_delay==0:
            self.target_qnet=self.target_soft_update(self.qnet, self.target_qnet, soft_tau)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net, soft_tau)
        # print('Time to update target: ', time.time() - start)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), path+'_q')
        torch.save(self.target_qnet.state_dict(), path+'_target_q')
        torch.save(self.policy_net.state_dict(), path+'_policy')

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load(path+'_q'))
        self.target_qnet.load_state_dict(torch.load(path+'_target_q'))
        self.policy_net.load_state_dict(torch.load(path+'_policy'))
        self.qnet.eval()
        self.target_qnet.eval()
        self.policy_net.eval()

def plot(rewards):
    plt.figure(figsize=(20,5))
    plt.plot(rewards)
    plt.savefig('rdpg.png')
    # plt.show()
    plt.clf()

class NormalizedActions(gym.ActionWrapper): # gym env wrapper
    def _action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        
        return action

    def _reverse_action(self, action):
        low  = self.action_space.low
        high = self.action_space.high
        
        action = 2 * (action - low) / (high - low) - 1
        action = np.clip(action, low, high)
        
        return action


if __name__ == '__main__':


    ENV = ['Pendulum', 'Reacher'][0]
    if ENV == 'Pendulum':
        env = gym.make("Pendulum-v1")
        # env = gym.make("Pendulum-v0")
        action_space = env.action_space
        state_space  = env.observation_space

    hidden_dim = 64
    explore_episodes = 10  # for random exploration
    batch_size = 8  # each sample in batch is an episode for lstm policy (normally it's timestep)
    update_itr = 1  # update iteration
    replay_buffer_size=1e6
    model_path='./model/rdpg'


    replay_buffer = ReplayBufferLSTM2(replay_buffer_size)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    torch.autograd.set_detect_anomaly(True)


    alg = RDPG(replay_buffer, state_space, action_space, hidden_dim)

    if args.train:
        # alg.load_model(model_path)

        # hyper-parameters
        max_episodes  = 1000
        max_steps   = 100
        # frame_idx   = 0
        rewards=[]

        for i_episode in range (max_episodes + explore_episodes):
            episode_start_time = time.time()
            episode_start_time = time.time()
            q_loss_list=[]
            policy_loss_list=[]
            state, _ = env.reset()
            episode_reward = 0
            last_action = env.action_space.sample()
            episode_state = []
            episode_action = []
            episode_last_action = []
            episode_reward = []
            episode_next_state = []
            episode_done = []
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            
            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out = alg.policy_net.get_action(state, last_action, hidden_in)
                next_state, reward, truncation, done, _,  = env.step(action)
                reward = reward.astype(np.float32)
                # if ENV !='Reacher':
                #     env.render()
                if step==0:
                    ini_hidden_in = hidden_in
                    ini_hidden_out = hidden_out
                episode_state.append(state)
                episode_action.append(action)
                episode_last_action.append(last_action)
                episode_reward.append(reward)
                episode_next_state.append(next_state)
                episode_done.append(done)  

                state = next_state
                last_action = action
                # frame_idx += 1
                if i_episode > explore_episodes:
                    if len(replay_buffer) > batch_size:
                        for _ in range(update_itr):
                            q_loss, policy_loss = alg.update(batch_size)
                            q_loss_list.append(q_loss)
                            policy_loss_list.append(policy_loss)
                        
                # if done:  # should not break for lstm cases to make every episode with same length
                #     break        

            if i_episode % 20 == 0:
                plot(rewards)
                alg.save_model(model_path)
            print('Eps: ', i_episode, '| Reward: ', np.sum(episode_reward), '| Loss: ', np.average(q_loss_list), np.average(policy_loss_list))
            print('Time for episode: ', time.time() - episode_start_time)
            replay_buffer.push(ini_hidden_in,
                               ini_hidden_out,
                               episode_state,
                               episode_action,
                               episode_last_action,
                               episode_reward,
                               episode_next_state,
                               episode_done)

            rewards.append(np.sum(episode_reward))
            alg.save_model(model_path)


    if args.test:
        test_episodes = 10
        max_steps=100
        alg.load_model(model_path)

        for i_episode in range (test_episodes):
            q_loss_list=[]
            policy_loss_list=[]
            state = env.reset()
            episode_reward = 0
            last_action = np.zeros(action_space.shape[0])
            hidden_out = (torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device), \
                torch.zeros([1, 1, hidden_dim], dtype=torch.float).to(device))  # initialize hidden state for lstm, (hidden, cell), each is (layer, batch, dim)
            
            for step in range(max_steps):
                hidden_in = hidden_out
                action, hidden_out= alg.policy_net.get_action(state, last_action, hidden_in, noise_scale=0.0)  # no noise for testing
                next_state, reward, done, _ = env.step(action)
                env.render()
                
                last_action = action
                state = next_state
                episode_reward += reward
                
                
                if done:
                    break
 
            print('Eps: ', i_episode, '| Reward: ', episode_reward)
            