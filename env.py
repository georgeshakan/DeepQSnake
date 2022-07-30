
import numpy as np 

class Env:
    def __init__(self,width,height):
        '''
        main methods: reset, step 
        reset: resets the environment so that snake is somewhat close to middle of screen
        step: takes an action and returns state, reward, done where
        state: is a height X width X 2 numpy array where the last
        channel is the previous screen and the first channel is the current screen
        reward: is 0, -1, or 1. It is -1 if the snake dies and 1 if it eats food.
        '''
        self.width = width
        self.height = height
        
        
        self.head = (-1,-1)


    def reset(self,random = True):
        '''
        resets the environment
        outputs state, reward, done (terminal state)
        state : current state (height,width,2) 
        reward: 0
        done: False
        '''
        
       
        self._create_snake(random = random)

        self._place_snake()
        self._replace_food()
        self._place_food()
        
        self.previous_screen = np.zeros((self.height,self.width))
        for u,v in self.snake:
            self.previous_screen[u,v-1] = 1
        f1,f2 = self.food 
        self.previous_screen[f1,f2] = 1

        return self._current_state(),0 , False

    

    def step(self,action):
        '''
        action: 0 = left, 1 = right, 2 = up, 3 = down
        it is a np.array of shape (4,)
        '''
        reward = -.05
        self.previous_screen = self.screen.copy() 

        direction = self._convert_action(action) 
        new_head = (self.snake[0][0] + direction[0],
        self.snake[0][1] + direction[1])
        
  
        done = self._is_terminal(new_head)
        if done:
            reward = -1

        
        if 0 <= new_head[0] < self.height and 0 <= new_head[1] < self.width:
            self.snake = [new_head] + self.snake 

        if new_head == self.food: 
            reward = 1
            self._replace_food()
        else:
            self.snake.pop() 

            

        self._place_snake()
        self._place_food()
        
        
        return self._current_state(),reward,done

            
        
        

        
    def _is_terminal(self,new_head):
        '''
        returns whether game is done or not
        '''
        if new_head in self.snake:
            done = True
        elif new_head[0] < 0 or new_head[0] >= self.height:
            done = True
        elif new_head[1] < 0 or new_head[1] >= self.width:
            done = True
        else:
            done = False
        return done 

    def _convert_action(self,action):
        '''
        converts the action to a direction
        '''
        if action == 0:
            return (0,-1)
        elif action == 1:
            return (0,1)
        elif action == 2:
            return (1,0)
        elif action == 3:
            return (-1,0)
        else:
            raise ValueError("action in env.step(action) must be 0,1,2,3")

    def _current_state(self):
        '''
        combines the current screen and the previous screen
        to out put a state'''
        return np.concatenate([np.expand_dims(self.screen,-1),
        np.expand_dims(self.previous_screen,axis = -1)],
        axis=-1)
    


    def _replace_food(self):
        '''
        resets self.food to a random location
        avoids placing it on the snake
        '''
        food_placed = False
        while not food_placed:
            row, col = np.random.randint(self.height) , np.random.randint(self.width)
            if self.screen[row,col] == 0:
                food_placed = True
                self.food = (row,col)

    def _place_food(self):
        '''
        places food on the screen
        '''
        f1,f2 = self.food
        self.screen[f1,f2] = 1          

    def _place_snake(self):
        '''
        uses self.snake to place the snake on the screen
        resets the screen and places food back 
        '''
        self.screen = np.zeros((self.height,self.width))
    
        for (u,v) in self.snake:
            self.screen[u,v] = 1
    
    def _create_snake(self,random = True):
        if not random:
            self.snake = [(self.height//2,self.width//2 +1 - j)
            for j in range(3) ]
        else:

            
            self.snake = []
            row, col = np.random.randint(self.height) , np.random.randint(self.width)
            self.snake.append((row,col))
            self.head = (row,col)
            

    
        
            
        
        
    
    

if __name__ == "__main__":
    env = Env(12,12)
    env.reset(random = False)

    
    
 
   
  
   
   

    
