
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