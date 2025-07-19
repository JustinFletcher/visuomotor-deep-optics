
import random
import torch

import pickle


class ReplayBufferWithHiddenStates:
    """
    A replay buffer that stores transitions including actor and critic hidden states.
    Each entry is a tuple:
    (actor_hidden, state, action, last_action, reward, last_reward, next_state, done)
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.advantageous = dict()

    def push(self, actor_hidden, state, action, last_action, reward, last_reward, next_state, done, advantageous=None):

        if len(self.buffer) < self.capacity:
            self.buffer.append(None)

        # print(f"buffer len {len(self.buffer)}")
        # print(f"buffer pos {self.position}")
        self.buffer[self.position] = (actor_hidden, state, action, last_action, reward, last_reward, next_state, done)
        if advantageous is not None:
            self.advantageous[self.position] = advantageous
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size, device=None):
        
        batch = random.sample(self.buffer, batch_size)
        actor_hidden_list = []
        s_lst, a_lst, la_lst, r_lst, lr_lst, ns_lst, d_lst = [], [], [], [], [], [], []
        for transition in batch:
            actor_hidden, state, action, last_action, reward, last_reward, next_state, done = transition
            actor_hidden_list.append(actor_hidden)
            s_lst.append(state)
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            lr_lst.append(last_reward)
            ns_lst.append(next_state)
            d_lst.append(done)

        # TODO: Speed up sampling by doing this parse only once.
        if device:
            actor_h = torch.cat([h.to(device) for (h, c) in actor_hidden_list], dim=0)
            actor_c = torch.cat([c.to(device) for (h, c) in actor_hidden_list], dim=0)
        else:
            actor_h = torch.cat([h for (h, c) in actor_hidden_list], dim=0)
            actor_c = torch.cat([c for (h, c) in actor_hidden_list], dim=0)
        actor_hidden_stacked = (actor_h, actor_c)

        return actor_hidden_stacked, s_lst, a_lst, la_lst, r_lst, lr_lst, ns_lst, d_lst

    def __len__(self):
        return len(self.buffer)
    
    def get_num_advantageous(self):
        """Get the number of True advantageous samples stored in the buffer."""
        return sum(1 for v in self.advantageous.values() if v)

    def get_num_disadvantageous(self):

        """Get the number of False advantageous samples stored in the buffer."""
        return sum(1 for v in self.advantageous.values() if not v)

    def save(self,
             dir_path,
             chunk_size=100):
        """Save the buffer in chunks to the specified directory using pickle."""
        import os
        os.makedirs(dir_path, exist_ok=True)
        meta = {
            'capacity': self.capacity,
            'position': self.position,
            'num_chunks': (len(self.buffer) + chunk_size - 1) // chunk_size,
            'advantageous': self.advantageous,
            'chunk_size': chunk_size
        }
        with open(os.path.join(dir_path, 'meta.pt'), 'wb') as f:
            pickle.dump(meta, f)

        for i in range(meta['num_chunks']):

            chunk = self.buffer[i * chunk_size:(i + 1) * chunk_size]

            with open(os.path.join(dir_path, f'chunk_{i:05d}.pt'), 'wb') as f:
                pickle.dump(chunk, f)

    def _load_single_buffer(self,
                            dir_path,
                            filter_disadvantageous=False,
                            explore_to_exploit_ratio=2):
        """Helper to load a single buffer chunk directory using pickle."""
        import os
        with open(os.path.join(dir_path, 'meta.pt'), 'rb') as f:
            meta = pickle.load(f)

        if not hasattr(self, 'capacity'):
            self.capacity = meta['capacity']

        # self.position = meta['position']
        self.advantageous = meta['advantageous']

        i = 0
        disadvantageous_budget = 0
        while True:
            chunk_path = os.path.join(dir_path, f'chunk_{i:05d}.pt')
            if not os.path.exists(chunk_path):
                break
            with open(chunk_path, 'rb') as f:
                chunk = pickle.load(f)

            # Filter the chunk based on the advantageous flag
            if filter_disadvantageous:
                filtered_chunk = list()
                for j, element in enumerate(chunk):

                    element_loc = (i * meta['chunk_size']) + j


                    if self.advantageous[element_loc]:
                        filtered_chunk.append(element)
                        disadvantageous_budget += explore_to_exploit_ratio

                    else:
                        if disadvantageous_budget > 0:
                            disadvantageous_budget -= 1
                            filtered_chunk.append(element)

                chunk = filtered_chunk
            self.buffer.extend(chunk)
            i += 1

    def restore(self,
                dir_path,
                filter_disadvantageous=False,
                explore_to_exploit_ratio=2):
        """Recursively load all buffer chunks from a directory tree."""
        import os
        self.buffer = []
        for root, dirs, files in os.walk(dir_path):
            if 'meta.pt' in files:
                try:
                    self._load_single_buffer(root,
                                            filter_disadvantageous,
                                            explore_to_exploit_ratio)
                except Exception as e:
                    print(f"Error loading buffer from {root}: {e}")
                    continue

        self.position = len(self.buffer) 


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

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, last_reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (hidden_in, hidden_out, state, action, last_action, reward, last_reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, lr_lst, ns_lst, hi_lst, ci_lst, ho_lst, co_lst, d_lst=[],[],[],[],[],[],[],[],[],[],[]
        batch = random.sample(self.buffer, batch_size)
        for sample in batch:
            (h_in, c_in), (h_out, c_out), state, action, last_action, reward, last_reward, next_state, done = sample
            s_lst.append(state) 
            a_lst.append(action)
            la_lst.append(last_action)
            r_lst.append(reward)
            lr_lst.append(last_reward)
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

        return hidden_in, hidden_out, s_lst, a_lst, la_lst, r_lst, lr_lst, ns_lst, d_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)
    


class ReplayBufferWithHiddenStatesBPTT:
    """
    A replay buffer for RNN training with BPTT. Stores sequences of transitions and the
    initial hidden state for each sequence. Supports advantageous filtering and saving/restoring.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.advantageous = dict()

    def push(self,
             initial_actor_hidden,
             initial_qf1_hidden,
             initial_qf2_hidden,
             observation_sequence,
             action_sequence,
             last_action_sequence,
             reward_sequence,
             last_reward_sequence,
             next_state_sequence,
             done_sequence,
             advantageous=None):
        """
        Add a full sequence of transitions with initial hidden state.
        Each sequence is a list of (state, action, last_action, reward, last_reward, next_state, done)
        """
        # Ensure that the observation_sequence and other sequences are of the same length
        assert len(observation_sequence) == len(action_sequence) == len(last_action_sequence) == \
               len(reward_sequence) == len(last_reward_sequence) == len(next_state_sequence) == \
               len(done_sequence), "All sequences must have the same length"

        # Ensure that initial hidden states are tuples of (h, c)
        assert isinstance(initial_actor_hidden, tuple) and len(initial_actor_hidden) == 2, \
            "Initial actor hidden state must be a tuple of (h, c)"
        assert isinstance(initial_qf1_hidden, tuple) and len(initial_qf1_hidden) == 2, \
            "Initial QF1 hidden state must be a tuple of (h, c)"
        assert isinstance(initial_qf2_hidden, tuple) and len(initial_qf2_hidden) == 2, \
            "Initial QF2 hidden state must be a tuple of (h, c)"
        
        # Print the shape of the initial hidden states for debugging
        # print(f"Initial actor hidden state shapes: {initial_actor_hidden[0].shape}, {initial_actor_hidden[1].shape}")
        # print(f"Initial QF1 hidden state shapes: {initial_qf1_hidden[0].shape}, {initial_qf1_hidden[1].shape}")
        # print(f"Initial QF2 hidden state shapes: {initial_qf2_hidden[0].shape}, {initial_qf2_hidden[1].shape}")

        # Ensure that the initial hidden states are tensors
        # assert isinstance(initial_actor_hidden[0], torch.Tensor) and isinstance(initial_actor_hidden[1], torch.Tensor), \
        #     "Initial actor hidden state must be tensors"
        # assert isinstance(initial_qf1_hidden[0], torch.Tensor) and isinstance(initial_qf1_hidden[1], torch.Tensor), \
        #     "Initial QF1 hidden state must be tensors"
        # assert isinstance(initial_qf2_hidden[0], torch.Tensor) and isinstance(initial_qf2_hidden[1], torch.Tensor), \
        #     "Initial QF2 hidden state must be tensors"

        # Store the shape of each sequence upon first push
        if len(self.buffer) == 0:
            self.sequence_length = len(observation_sequence)
            self.observation_shape = observation_sequence[0].shape if observation_sequence else None
            self.action_shape = action_sequence[0].shape if action_sequence else None
            self.last_action_shape = last_action_sequence[0].shape if last_action_sequence else None
            self.reward_shape = reward_sequence[0].shape if reward_sequence else None
            self.last_reward_shape = last_reward_sequence[0].shape if last_reward_sequence else None
            self.next_state_shape = next_state_sequence[0].shape if next_state_sequence else None

        # Check that all sequences have the same shape as the first push
        assert all(obs.shape == self.observation_shape for obs in observation_sequence), \
            "All observations must have the same shape"
        assert all(act.shape == self.action_shape for act in action_sequence), \
            "All actions must have the same shape"
        assert all(last_act.shape == self.last_action_shape for last_act in last_action_sequence), \
            "All last actions must have the same shape"
        assert all(rew.shape == self.reward_shape for rew in reward_sequence), \
            "All rewards must have the same shape"
        assert all(last_rew.shape == self.last_reward_shape for last_rew in last_reward_sequence), \
            "All last rewards must have the same shape"
        assert all(next_state.shape == self.next_state_shape for next_state in next_state_sequence), \
            "All next states must have the same shape"

        # Ensure that the sequence length matches the stored sequence length
        assert len(observation_sequence) == self.sequence_length, \
            f"Sequence length must be {self.sequence_length}, got {len(observation_sequence)}"
        
        view_shapes = False
        if view_shapes:
            # Print shapes of sequences for debugging
            print(f"Observation sequence shape: {[obs.shape for obs in observation_sequence]}")
            print(f"Action sequence shape: {[act.shape for act in action_sequence]}")
            print(f"Last action sequence shape: {[last_act.shape for last_act in last_action_sequence]}")
            print(f"Reward sequence shape: {[rew.shape for rew in reward_sequence]}")
            print(f"Last reward sequence shape: {[last_rew.shape for last_rew in last_reward_sequence]}")
            print(f"Next state sequence shape: {[next_state.shape for next_state in next_state_sequence]}")

        # Add the sequence to the buffer
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (initial_actor_hidden,
                                      initial_qf1_hidden,
                                      initial_qf2_hidden,     
                                      observation_sequence,           
                                      action_sequence,
                                      last_action_sequence,
                                      reward_sequence,
                                      last_reward_sequence,
                                      next_state_sequence,
                                      done_sequence)
        if advantageous is not None:
            self.advantageous[self.position] = advantageous
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, device=None):
        """
        Sample a batch of sequences. Returns tuple:
        (batch of initial_hidden_states, batch of sequences)
        """
        batch = random.sample(self.buffer, batch_size)
        (initial_actor_hidden,
         initial_qf1_hidden,
         initial_qf2_hidden,
         observation_sequence,                
         action_sequence,
         last_action_sequence,
         reward_sequence,
         last_reward_sequence,
         next_state_sequence,
         done_sequence) = zip(*batch)
        if device:
            initial_actor_h0 = [h[0].to(device) for h in initial_actor_hidden]
            initial_actor_c0 = [h[1].to(device) for h in initial_actor_hidden]
            initial_qf1_h0 = [h[0].to(device) for h in initial_qf1_hidden]
            initial_qf1_c0 = [h[1].to(device) for h in initial_qf1_hidden]
            initial_qf2_h0 = [h[0].to(device) for h in initial_qf2_hidden]
            initial_qf2_c0 = [h[1].to(device) for h in initial_qf2_hidden]
        return (initial_actor_h0, initial_actor_c0), (initial_qf1_h0, initial_qf1_c0), (initial_qf2_h0, initial_qf2_c0), observation_sequence, action_sequence, last_action_sequence, reward_sequence, last_reward_sequence, next_state_sequence, done_sequence

    def __len__(self):
        return len(self.buffer)

    def get_num_advantageous(self):
        return sum(1 for v in self.advantageous.values() if v)

    def get_num_disadvantageous(self):
        return sum(1 for v in self.advantageous.values() if not v)

    def save(self, dir_path, chunk_size=100):
        import os, pickle
        os.makedirs(dir_path, exist_ok=True)
        meta = {
            'capacity': self.capacity,
            'position': self.position,
            'num_chunks': (len(self.buffer) + chunk_size - 1) // chunk_size,
            'advantageous': self.advantageous,
            'chunk_size': chunk_size
        }
        with open(os.path.join(dir_path, 'meta.pt'), 'wb') as f:
            pickle.dump(meta, f)

        for i in range(meta['num_chunks']):
            chunk = self.buffer[i * chunk_size:(i + 1) * chunk_size]
            with open(os.path.join(dir_path, f'chunk_{i:05d}.pt'), 'wb') as f:
                pickle.dump(chunk, f)

    def _load_single_buffer(self, dir_path, filter_disadvantageous=False, explore_to_exploit_ratio=2):
        import os, pickle
        with open(os.path.join(dir_path, 'meta.pt'), 'rb') as f:
            meta = pickle.load(f)

        if not hasattr(self, 'capacity'):
            self.capacity = meta['capacity']

        self.advantageous = meta['advantageous']
        i = 0
        disadvantageous_budget = 0
        while True:
            chunk_path = os.path.join(dir_path, f'chunk_{i:05d}.pt')
            if not os.path.exists(chunk_path):
                break
            with open(chunk_path, 'rb') as f:
                chunk = pickle.load(f)

            if filter_disadvantageous:
                filtered_chunk = []
                for j, element in enumerate(chunk):
                    element_loc = (i * meta['chunk_size']) + j
                    if self.advantageous.get(element_loc, True):
                        filtered_chunk.append(element)
                        disadvantageous_budget += explore_to_exploit_ratio
                    elif disadvantageous_budget > 0:
                        disadvantageous_budget -= 1
                        filtered_chunk.append(element)
                chunk = filtered_chunk
            self.buffer.extend(chunk)
            i += 1

    def restore(self, dir_path, filter_disadvantageous=False, explore_to_exploit_ratio=2):
        import os
        self.buffer = []
        for root, dirs, files in os.walk(dir_path):
            if 'meta.pt' in files:
                try:
                    self._load_single_buffer(root, filter_disadvantageous, explore_to_exploit_ratio)
                except Exception as e:
                    print(f"Error loading buffer from {root}: {e}")
        self.position = len(self.buffer)