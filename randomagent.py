
import numpy as np

class randomAgent:
    def __init__(self):
        self.actions = [0,1,2,3]

    def act(self, state):
        return np.random.choice(self.actions)