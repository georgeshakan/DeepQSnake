import numpy as np 

class ExperienceReplay:
    '''
    create a working memory of past experience 
    
    memory has size max_size

    Experience Replay remembers state, new state, action, reward, and done

    methods: store_transition, sample_memory
    '''

    
    def __init__(self,max_size, input_shape):

        #max size of memory
        self.mem_size = max_size 
        self.memory_counter = 0 

        #rows of all future np arrays are indexed by the memory
        #inputsize for snake is (screen_width , screen_height , 3) 

        #np.float32 works well with torch
        self.state_memory = np.zeros((self.mem_size, *input_shape) , dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape) , dtype = np.float32)

        ##size (mem_size,)
        self.action_memory = np.zeros(self.mem_size, dtype = np.int64 )
        self.reward_memory = np.zeros(self.mem_size , dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size , dtype = np.bool_)

    def store_transition(self,state, action, reward, state_, done):
        '''
        stores inputs into memory at index self.memory_counter
        and then increments memory_counter
        '''
        #cycle through the memory 
        ind = self.memory_counter % self.mem_size
        


        self.state_memory[ind] = state
        self.action_memory[ind] = action
        self.reward_memory[ind] = reward
        self.new_state_memory[ind] = state_
        self.terminal_memory[ind] = done 

        self.memory_counter +=1

    def sample_memory(self, batch_size):
        '''
        sample batch_size elements from memory
        '''

        max_mem = min(self.memory_counter,self.mem_size)

        #returns a random batch of indices in [0,max_mem)

        batch = np.random.choice(max_mem, size = batch_size, replace = False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions , rewards , states_ ,  dones