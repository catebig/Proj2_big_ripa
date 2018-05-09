import Module_base as M
import numpy as np
import torch

# Test Lienar case


# Create List of Modules (first attempt of NN structure)
listModules = []
number_of_classes = 2
number_of_inputs = 2
nb_train_input = 200 
nb_test_input = 10

listModules.append(M.Linear(number_of_classes,number_of_inputs))
listModules.append(M.Sigmoid())
# Give the list as input to a sequential Module
loss = M.LossMSE()
NN = M.Sequential(listModules,loss,eta=0.1)

#y = torch.randn(Nsample,number_of_classes)
train_input = torch.randn(nb_train_input,number_of_inputs)
train_target = torch.zeros(nb_train_input,number_of_classes)
for i in range(nb_train_input):
    if train_input[i,0]+train_input[i,1]>0.25: 
        train_target[i,0] = 0.21
        train_target[i,1] = 0.81
    else: 
        train_target[i,0] = 0.75
        train_target[i,1] = 0.25


test_input = torch.randn(nb_test_input,number_of_inputs)
test_target = torch.zeros(nb_test_input,number_of_classes)
for i in range(nb_test_input):
    if test_input[i,0]+test_input[i,1]>0: 
        test_target[i,0] = 0.21
        test_target[i,1] = 0.81
    else: 
        test_target[i,0] = 0.75
        test_target[i,1] = 0.25


nb_epochs = 1000 

for k in range(0, nb_epochs):

    nb_train_errors = 0

    pred = NN.forward(train_input) 
    loss_output = NN.loss.forward(pred,train_target)
    NN.backward()
    NN.param()
    
    nb_train_errors = sum(abs(train_target.max(1)[1] - pred.max(1)[1]))


    pred_test = NN.forward(test_input) 
    nb_test_errors = sum(abs(test_target.max(1)[1] - pred_test.max(1)[1]))
   
    print('{:d} loss: {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  sum(loss_output),
                  (100 * nb_train_errors) / nb_train_input,
                  (100 * nb_test_errors) / nb_test_input))







# def nearest_classification(train_input, train_target, x):
#     dist = (train_input - x).pow(2).sum(1).view(-1)
#     _, n = torch.min(dist, 0)
#     return train_target[n[0]]

# def compute_nb_errors(train_input, train_target,
#                       test_input, test_target,
#                       mean = None, proj = None):

#     if mean is not None:
#         train_input = train_input - mean
#         test_input = test_input - mean

#     if proj is not None:
#         train_input = train_input.mm(proj.t())
#         test_input = test_input.mm(proj.t())

#     nb_errors = 0

#     # With loop, but I prefer clearer code when counting errors
#     for n in range(0, test_input.size(0)):
#         if test_target[n] != nearest_classification(train_input, train_target, test_input[n]):
#             nb_errors = nb_errors + 1

#     return nb_errors