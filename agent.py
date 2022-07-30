

from experiencereplay import ExperienceReplay
from network import Network
from metrics import Metrics
import numpy as np 
import torch 






class Agent:
    def __init__(self,gamma,lr,width,height,eps=1.0,
                replace =1000,max_memory = 10_000,anneal = 50_000,dev = torch.device("cpu"),debug = False):
        self.eps = eps
        self.gamma = gamma
        self.width = width
        self.height = height
        self.memory = ExperienceReplay(max_memory,(width,height,2))
        self.replace_target_count = replace
        input_shape = (width,height,2)
        self.Q = Network(input_shape,dev = dev)
        self.Q_next = Network(input_shape,dev = dev)
        self.actions = [0,1,2,3]
        self.batch_size = 32
        self.loss = torch.nn.MSELoss().to(dev)
        self.learn_step_counter = 0
        self.optimizer = torch.optim.Adam(self.Q.parameters(),lr = lr)
        self.anneal_rate = 1.0/anneal
        self.Metrics = Metrics()
       

       
    def act(self,state):
        '''
        choose action for agent during
        evaluation
        '''
        state_tensor = torch.unsqueeze(
            torch.tensor(state,
            dtype = torch.float32),dim = 0)
        state_tensor.to(self.Q.device)
        output = torch.argmax(self.Q(state_tensor)).item()
      
        return output
        

    def choose_action(self,state):
        '''
        choose action for agent during
        training'''
        if np.random.random() < self.eps:
            return np.random.choice(self.actions)
        else:
            
            state_tensor = torch.unsqueeze(
                torch.tensor(state,
                dtype = torch.float32),dim = 0)
            state_tensor.to(self.Q.device)
            self.Metrics.update_action_history(
                self.Q(state_tensor).detach().to(torch.device('cpu'))
            )
            self.logger.debug(f"chosen action: {torch.argmax(self.Q(state_tensor)).item()}")
            
            return torch.argmax(self.Q(state_tensor)).item()
    def store_transition(self,state,action,reward,state_,done):
        self.memory.store_transition(state,action,reward,state_,done)
    
    def sample_memory(self):
        state , action , reward, new_state , done = \
            self.memory.sample_memory(self.batch_size)
        states = torch.tensor(state).to(self.Q.device)
        rewards = torch.tensor(reward).to(self.Q.device)
        actions = torch.tensor(action).to(self.Q.device)
        states_ = torch.tensor(new_state).to(self.Q.device)
        dones = torch.tensor(done,dtype = torch.bool).to(self.Q.device)

        return states, actions, rewards, states_ , dones

    def replace_target_network(self):
        self.logger.debug("Updating target network")
        if self.learn_step_counter % self.replace_target_count == 0:
            self.Q_next.load_state_dict(self.Q.state_dict())
        
    def epsilon_decrement(self):
        
        self.eps = self.eps - self.anneal_rate \
                    if self.eps > .1 else .1

    def load(self,filename):
   
        self.Q.load_state_dict(torch.load(filename,map_location=self.Q.device))
        self.Q_next.load_state_dict(torch.load(filename,map_location=self.Q.device))

    def save(self,filename):
        torch.save(self.Q.state_dict(),filename)

        
    def learn(self):
        '''
        samples from memory and updates Q network
        via 
        Q(s,a) = r + gamma max_a Q_next(s_,a)
        '''
        if self.memory.memory_counter < self.batch_size:
            return None 
        self.optimizer.zero_grad()
        self.logger.debug(f"Learning{self.learn_step_counter}")
        self.replace_target_network()
        
        ##sampling from memory logic
        state, action, reward, new_state, done = \
            self.sample_memory()
        self.logger.debug(f"Samples: {state.shape}")
        self.logger.debug(state[0,:,:,0])
        self.logger.debug(state[0,:,:,1])
        self.logger.debug(
        f"Sample difference: {torch.sum((state[0,:,:,0] - state[0,:,:,1])**2)}")
        indices = np.arange(self.batch_size)

        #q_next is the the value of next state
        #from old network
        q_next = self.Q_next(new_state).max(dim=1)[0]
        
        
        q_next[done] = 0.0

        q_pred = self.Q(state)[indices,action].to(self.Q.device)
        q_target = (reward + self.gamma * q_next).to(self.Q.device).detach()

        self.Metrics.update_Q_history(
            torch.mean(q_pred.detach().to(torch.device('cpu'))).item())
        self.Metrics.update_Q_next_history(
            torch.mean(q_target.detach().to(torch.device('cpu'))).item())
            
        self.logger.debug(f'done: {done}')
        self.logger.debug(f'reward: {reward}')
        self.logger.debug(f"q_next: {q_target}")
        self.logger.debug(f"q_pred: {q_pred}")
        for u in self.Q.parameters():
            self.logger.debug(f"Before update: {torch.var(u).detach().item()}")
        loss = self.loss(q_pred,q_target).to(self.Q.device)
        
        loss.backward()
        self.optimizer.step()
        for u in self.Q.parameters():
            self.logger.debug(f"After update: {torch.var(u).detach().item()}")
        self.learn_step_counter +=1
        self.epsilon_decrement()
        self.Metrics.update_loss_history(loss.item())
        self.Metrics.update_epsilon_history(self.eps)


if __name__ == "__main__":
    agent = Agent(0.99,0.01,30,30)
    state = np.ones((30,30,2))
    actions = 2
    rewards = 1
    dones =False
    for _ in range(50):
        agent.store_transition(state,actions,rewards,state,dones)
    agent.learn()
    state, action, reward, new_state, done = \
            agent.sample_memory()
    agent.optimizer.zero_grad()
    indices = np.arange(agent.batch_size)
    q_pred = agent.Q(state)[indices,action]
    q_next = agent.Q_next(new_state).max(dim=1)[0]

    q_next[done] = 0.0
    q_target = reward + agent.gamma * q_next
    loss = agent.loss(q_pred,q_target).to(agent.Q.device)
    loss.backward()
    agent.optimizer.step()

