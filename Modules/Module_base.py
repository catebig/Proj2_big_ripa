import torch
import numpy as np
from abc import ABC, abstractmethod


# Parent Module class:
# Define the structure of a general Module object
class Module(object):

  def forward(self, *input):
    raise NotImplementedError

  @abstractmethod
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
    self.bias = torch.randn(*dim[:-1],1)

  def forward(self, *input):
    self.input = input[0]
    return torch.mm(self.weights,input[0])+self.bias

  def backward(self, *gradwrtoutput):
    raise NotImplementedError

  def param(self):
      return []



class LossMSE(Module):

  def forward(self, *input):
    self.diff = input[0] - input[1]
    return torch.mm(torch.t(self.diff),self.diff)/torch.numel(self.diff)

  def backward(self, *gradwrtoutput):
    return 2*self.diff/torch.numel(self.diff)

  def param(self):
    return []





class Sequential(Module):

  def __init__(self,listModules):
    self.graphModules = listModules

  def forward(self, *input):
    x = input[0]
    for Module in self.graphModules:
      x = Module.forward(x)
    return x

  def backward(self):
     raise NotImplementedError
  

  def param(self):
      return []
