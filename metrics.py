


class Metrics:

    def __init__(self):

        self.loss_history = [] 
        self.reward_history = []
        self.epsilon_history = [] 
        self.action_history = [] 
        self.Q_next_history = [] 
        self.Q_history = [] 

    def update_Q_next_history(self,Q_next):
        self.Q_next_history.append(Q_next)
    
    def update_Q_history(self,Q):
        self.Q_history.append(Q)

    def update_action_history(self,action):
        self.action_history.append(action)

    def update_loss_history(self,loss):
        self.loss_history.append(loss)
    
    def update_reward_history(self,reward):
        self.reward_history.append(reward)
    
    def update_epsilon_history(self,epsilon):
        self.epsilon_history.append(epsilon)