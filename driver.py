import Modules.Module_base as M
import numpy as np
import torch

# Test Lienar case



# Create List of Modules (first attempt of NN structure)
listModules = []
number_of_outputs = 2
number_of_inputs = 2
listModules.append(M.Linear(number_of_outputs,number_of_inputs))
listModules.append(M.Sigmoid())
# Give the list as input to a sequential Module
loss = M.LossMSE()
NN = M.Sequential(listModules,loss)

loss = M.LossMSE()
y = torch.randn(10,number_of_outputs)
x = torch.randn(10,number_of_inputs)
c = NN.forward(x)
r = loss.forward(c,y)
print(c)
#print(r)