import Modules.Module_base as M
import numpy as np
import torch

# Test Lienar case



# Create List of Modules (first attempt of NN structure)
listModules = []
listModules.append(M.Linear(5,5))
listModules.append(M.Sigmoid())
# Give the list as input to a sequential Module
NN = M.Sequential(listModules)

loss = M.LossMSE()
y = torch.randn(5,1)
x = torch.randn(5,1)
c = NN.forward(x)
r = loss.forward(c,y)
print(c)
print(r)