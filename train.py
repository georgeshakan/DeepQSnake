from agent import Agent
from env import Env 
import numpy as np
import torch 
import os




class Train:
    def __init__(self,gamma,lr,width,height,eps = 1.0,replace = 10_000,
    max_memory = 1_000_000,anneal = 50_000,dev = torch.device("cpu"),
    debug = False,warmup = False,load_file = None):

        
        self.agent = Agent(gamma,lr,width,height,eps = eps,replace = replace,max_memory =max_memory,anneal = anneal,dev = dev,debug = debug)
        self.env = Env(width,height)
        if load_file is not None:
            self.agent.load(load_file)

        self.score_history = []
        self.width = width
        self.height = height
        self.max_memory = max_memory
        self.life_history = []
        if warmup:
            self._warmup()
        
    def train(self,num_steps):
        for i in range(num_steps):
            self.train_epsiode()
            if i % 1_000 == 0 and i > 0:
                print(f"Step: {i}")
                print(f'eps: {self.agent.eps}')
                print(f'loss: {np.mean(self.agent.Metrics.loss_history[-1000:])}')
                print(f'scores: {np.mean(self.score_history[-1000:])}')
                print(f'time alive: {np.mean(self.life_history[-1000:])}')
                action_vals = torch.zeros((1,4))
                for u in self.agent.Metrics.action_history[-1000:]:
                    action_vals += u
                
                print(f"Average action values:{action_vals/1000}")
                print(f"Q_pred:{np.mean(self.agent.Metrics.Q_history[-1000:])}")
                print(f"Q_target:{np.mean(self.agent.Metrics.Q_next_history[-1000:])}")
                print("--------------------------------------------")
                if i %1_000 == 0:
                    self.agent.save(f"width{self.width}height{self.height}")

    def train_epsiode(self,warmup = False):
        state,_,done = self.env.reset()
        score = 0
        life = 0
       
        while not done:
            action = self.agent.choose_action(state)
            state_,reward,done = self.env.step(action) 

            self.agent.store_transition(state,action,reward,state_,done)
            if not warmup:
                self.agent.learn()
            score +=reward
            
            life +=1
            state = state_
        self.life_history.append(life)
        self.score_history.append(score)

    def _warmup(self):
        for _ in range(self.max_memory//60):
            self.train_epsiode(warmup = True)


    def load(self,filename):
        self.agent.load(filename)











if __name__ == "__main__":

    train = Train(.99,.1,15,15,replace = 10,max_memory = 10_000,anneal = 500_000, debug = True )
    #train.train(21)