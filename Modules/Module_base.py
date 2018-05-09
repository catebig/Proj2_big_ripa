import torch
import numpy as np
from abc import ABC, abstractmethod


# Parent Module class:
# Define the structure of a general Module object
class Module(object):

  def forward(self, *input):
    raise NotImplementedError

  def backward(self, *gradwrtoutput):
    raise NotImplementedError

  def param(self):
      return []

class Sigmoid(Module):

  def forward(self, *input):
    self.input = input[0]
    return 1/(1+np.exp(-input[0]))

  def backward(self, *gradwrtoutput):
    raise NotImplementedError

  def param(self):
      return []

class Linear(Module):

  def __init__(self, *dim):
    self.weights = torch.randn(*dim)
    self.bias = torch.randn(1,*dim[:-1])
    self.gradwrtweights = torch.zeros(*dim)
    self.gradwrtbias = torch.zeros(1,*dim[:-1])

  def forward(self, *input):
    self.input = input[0]
    return torch.mm(self.input,self.weights.t())+self.bias

  def backward(self, gradwrtoutput):
    # Compute gradient with respect to input
    gradwrtinput = gradwrtoutput * self.weights
    # Compute derivatives of loss wrt parameters
    self.gradwrtweights = 
    self.gradwrtbias = torch.mean(gradwrtinput,0)

    return gradwrtinput

  def param(self):
      return []



class LossMSE(Module):

  def forward(self, *input):
    self.diff = input[0] - input[1]
    loss = torch.sum(torch.pow(self.diff,2),1)/self.diff.size(0)
    return loss 

  def backward(self, *gradwrtoutput):
    return 2*self.diff/self.diff.size(0)

  def param(self):
    return []





class Sequential(Module):

  def __init__(self,listModules,loss):
    self.graphModules = listModules
    self.loss = loss

  def forward(self, *input):
    x = input[0]
    for Module in self.graphModules:
      x = Module.forward(x)
    return x
  
  def backward(self):
     raise NotImplementedError
  

  def param(self):
      return []
