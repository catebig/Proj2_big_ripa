import torch
import numpy as np
#from abc import ABC, abstractmethod


# Parent Module class:
# Define the structure of a general Module object
class Module(object):

  def forward(self, *input):
    raise NotImplementedError

  def backward(self, *gradwrtoutput):
    raise NotImplementedError

  def param(self,eta):
      return []




class Sigmoid(Module):

  def forward(self, input):
    self.input = input
    return 1/(1+np.exp(-self.input))

  def backward(self, gradwrtoutput):
    return 1+np.exp(-gradwrtoutput)

  def param(self,eta):
      return []

class Linear(Module):

  def __init__(self, *dim):
    self.weights = torch.randn(*dim)
    self.bias = torch.randn(1,*dim[:-1])
    self.gradwrtweights = torch.zeros(*dim)
    self.gradwrtbias = torch.zeros(1,*dim[:-1])

  def forward(self, input):
    self.input = input
    return torch.mm(self.input,self.weights.t())+self.bias

  def backward(self, gradwrtoutput):
    # Remark: gradwrtoutput is dl/ds^(l+1) - should be 10x2
    # Compute gradient with respect to input: dl/dx = w^(l+1)^T * dl/ds^(l+1)
    gradwrtinput = torch.mm(gradwrtoutput,self.weights) # weights 2x2 # should be 10x2 (same size as input x)
    # Compute derivatives of loss wrt parameters: dl/dx^(l) = dl/ds^(l)*x^(l-1)^T
    self.gradwrtweights = torch.mm(self.input.t(),gradwrtoutput)
    self.gradwrtbias = torch.mean(gradwrtoutput,0) #dl/db^(l) = dl/dx^(l) (in the linear case) we take the mean over the samples

    return gradwrtinput

  def param(self,eta):
      # Update the parameters
      self.weights = self.weights-eta*self.gradwrtweights
      self.bias = self.bias-eta*self.gradwrtbias

      return []



class LossMSE(Module):

  def forward(self, *input):
    self.diff = input[0] - input[1]
    loss = torch.sum(torch.pow(self.diff,2),1)/self.diff.size(0)
    return loss 

  def backward(self):
    return -2.*self.diff/self.diff.size(0)

  def param(self,eta):
    return []





class Sequential(Module):

  def __init__(self,listModules,loss,eta):
    self.graphModules = listModules
    self.loss = loss
    self.eta = eta

  def forward(self, input):
    x = input
    for m in self.graphModules:
      x = m.forward(x)
    return x
  
  def backward(self):
    # Grad_1 l(x^L)
    dldx = self.loss.backward()
    for m in reversed(self.graphModules):
      dldx = m.backward(dldx)
    return []

  def param(self):
    for m in self.graphModules:
      m.param(self.eta)
    return []
