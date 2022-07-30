from env import Env
import pygame, sys
from agent import Agent
import os 



def run_game():
    '''
    script to run game
    '''

    width = 10
    height = 10
    block_size = 20
    screen_width,screen_height = block_size*width,block_size*height
    
    
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((screen_width,screen_height))
    #(self,gamma,lr,width,height,
    #            replace =1000,max_memory = 10_000,anneal = 50_000,dev = torch.device("cpu"))
    env = Env(width,height)
    agent = Agent(.99,3e-4,width,height)
    PATH = os.path.join("models",f'width{width}height{height}')
    agent.load(PATH)
    done = True 
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if done:
            env.reset(random = True)
            
            

        
        screen.fill("White")
        for row in range(height):
            for col in range(width):
                if env.screen[row][col] == 1:
                    color = (0,0,0)
                    if (row,col) == env.food:
                        color = (255,0,0)
                    
                    pygame.draw.rect(screen,color,
                    (col*block_size,row*block_size,
                    block_size,block_size))

        
        action = agent.act(env._current_state())
        
        _, reward , done = env.step(action)
        
        




        pygame.display.update()
        clock.tick(10)



if __name__ == "__main__":
    # "randomAI"
    # "human"
    # "dqnAI"
    run_game()