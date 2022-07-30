import numpy as np
import torch.nn as nn 
import torch.nn.functional as F 
import torch

class Network(nn.Module):

    def __init__(self,input_shape,dev =torch.device("cpu") ):
        super(Network,self).__init__()
        hidden_layers = 512

        

        self.conv1 = nn.Conv2d(in_channels = 2,out_channels = 64,kernel_size = 3,stride = 1,padding = 'same')
        self.conv2 = nn.Conv2d(in_channels =64,out_channels = 64,kernel_size = 3,stride = 1,padding = 'same')
        input_size = self._compute_shape(input_shape)
        
        self.fc1 = nn.Linear(input_size, hidden_layers)
        self.fc2 = nn.Linear(hidden_layers, 4)
        self.device = dev 
        self.to(dev)

    def _compute_shape(self,input_shape):
        x = torch.unsqueeze(torch.zeros(input_shape),dim =0)
        x = torch.transpose(x,-1,-3)
        x = torch.transpose(x,-1,-2)
        x  = F.relu(self.conv1(x))
    
    
        x = F.relu(self.conv2(x))

        x = torch.flatten(x,start_dim = 1)
        return x.reshape(-1).shape[0]

    def forward(self,x):
        x = x.to(self.device)
        x = torch.transpose(x,-1,-3)
        x = torch.transpose(x,-1,-2)
        x  = F.relu(self.conv1(x))
 
        x = F.relu(self.conv2(x))
  
        x = torch.flatten(x,start_dim = 1)
        x = F.relu(self.fc1(x))
        
        return self.fc2(x)
        

if __name__ == "__main__":

    network = Network((10,10,2))
    x = torch.ones((1,10,10,2))
    y = network(x)
   
