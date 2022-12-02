import numpy as np
import random
from collections import namedtuple, deque
from queue import PriorityQueue

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle 
from os.path import exists

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 8         # minibatch size
# GAMMA = 0.99            # discount factor
GAMMA = 0.98            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 8        # how often to update the network

MAX_PRIORITY = -1000

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PriorityAgent():
    """Interacts with and learns from the environment."""

    def load(self, agent_i):        
        floc = 'checkpoint_local_{}.pth'.format(agent_i)
        if exists(floc):
            self.qnetwork_local.load_state_dict(torch.load(floc))
        ftarget = 'checkpoint_target_{}.pth'.format(agent_i)
        if exists(ftarget):
            self.qnetwork_target.load_state_dict(torch.load(ftarget))            
        self.memory.load(agent_i)
        
    def save(self, agent_i):
        torch.save(self.qnetwork_local.state_dict(), 'checkpoint_local_{}.pth'.format(agent_i))
        torch.save(self.qnetwork_target.state_dict(), 'checkpoint_target_{}.pth'.format(agent_i))         
        self.memory.save(agent_i)
        
    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = PriorityReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences,expz = self.memory.sample()
                td_errors = self.learn(experiences, GAMMA)
                for i,exp in enumerate(expz):
                    # This might have problems!
                    self.memory.addExp(td_errors[i],exp)                    

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # print("state:{}".format(state))
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"        
        
        
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        TD_errors = Q_targets - Q_expected
        
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        # Should be 
        return TD_errors

        
# Algorithm 1 Double DQN with proportional prioritization
# 1: Input: minibatch k, step-size η, replay period K and size N, exponents α and β, budget T.
# 2: Initialize replay memory H = ∅, ∆ = 0, p1 = 1
# 3: Observe S0 and choose A0 ∼ πθ(S0)
# 4: for t = 1 to T do
# 5: Observe St, Rt, γt
# 6: Store transition (St−1, At−1, Rt, γt, St) in H with maximal priority pt = maxi<t pi
# 7: if t ≡ 0 mod K then
# 8: for j = 1 to k do
# 9: Sample transition j ∼ P(j) = p_α_j / Sum_i( p_a_i )
# 10: Compute importance-sampling weight wj = (N · P(j))−β / max_i wi
# 11: Compute TD-error δj = Rj + γj * Qtarget(Sj , arg maxa Q(Sj , a)) − Q(Sj−1, Aj−1)
# 12: Update transition priority pj ← |δj |
# 13: Accumulate weight-change ∆ ← ∆ + wj · δj · ∇θQ(Sj−1, Aj−1)
# 14: end for
# 15: Update weights θ ← θ + η · ∆, reset ∆ = 0
# 16: From time to time copy weights into target network θtarget ← θ
# 17: end if
# 18: Choose action At ∼ πθ(St)
# 19: end for        
    def learn_double_priority(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        "*** YOUR CODE HERE ***"        
        
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)           
        
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])         

class PriorityReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = SortedDict()  
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = Experience
        self.seed = random.seed(seed)
        self.size = 0
        self.fullValue = 0

#     def load(self,i):
#         fmem = 'memory_{}.mem'.format(i)
#         if exists(fmem):
#             filehandler = open(fmem, 'rb') 
#             self.memory = pickle.load(filehandler)        
        
#     def save(self,i):
#         fmem = 'memory_{}.mem'.format(i)
#         filehandler = open(fmem, 'wb')
#         pickle.dump(self.memory, filehandler)
    
    def remove(self, p):
        v = None
        if self.memory[p]:       # Not empty?
            v = self.memory[p].pop()
            self.fullValue -= p
            self.size -= 1
        if not self.memory[p]:   # empty?
            del self.memory[p]
        return v
    
    def addExp(self, p , exp):
        if self.memory not in p:
            self.memory[p] = deque()
        self.memory[p].appendleft(e)
        self.size += 1
        self.fullValue += p
        
        if self.size > self.buffer_size:
# Smallest first
            for mf in self.memory:
                self.remove(mf)
                break
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        
        e = self.experience(state, action, reward, next_state, done)
        self.addExp(MAX_PRIORITY,e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        
        experiences = []
        for i in range(0,self.batch_size):            
            pi = random.random() * self.fullValue
            pc = 0
            for mf in self.memory:
                pc += mf * len(self.memory[mf])
                if pi <= pc:
                    experiences.append(self.remove(mf))
                    break
                    
        # experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones),experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
            
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = Experience
        self.seed = random.seed(seed)

    def load(self,i):
        fmem = 'memory_{}.mem'.format(i)
        if exists(fmem):
            filehandler = open(fmem, 'rb') 
            self.memory = pickle.load(filehandler)        
        
    def save(self,i):
        fmem = 'memory_{}.mem'.format(i)
        filehandler = open(fmem, 'wb')
        pickle.dump(self.memory, filehandler)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)