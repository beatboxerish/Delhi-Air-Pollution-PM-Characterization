#!/usr/bin/env python
# coding: utf-8

### libraries and modules
from datetime import datetime
import matplotlib.pyplot as plt
import tqdm.notebook as tqdm
import pandas as pd
import numpy as np
import pickle
import copy

import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.models import ApproximateGP
from torch.optim.lr_scheduler import StepLR




### defining functions and classes
def get_kernel(kernel = 0, ard = None):
    """
    Returns the kernel according to the index given as an argument
    Ard here allows us to fit a separate lengthscale to each dimension/features. Lengthscale is one of the hyperparameters that 
    learn for the kernel function.
    """
    if kernel == 0:
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = ard))
    elif kernel == 1:
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 0.5, ard_num_dims = ard))
    elif kernel == 2:
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 1.5, ard_num_dims = ard))
    elif kernel == 3:
        return gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 2.5, ard_num_dims = ard))
    elif kernel == 4:
        k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(active_dims = [2]))
        k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 1.5, active_dims = [0,1]))
        return gpytorch.kernels.ProductKernel(k1, k2)
    elif kernel == 5:
        k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(active_dims = [2]))
        k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 1.5, active_dims = [0,1]))
        return gpytorch.kernels.AdditiveKernel(k1, k2)
    elif kernel == 6:
        k1 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(active_dims = [2]))
        k2 = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu = 0.5, active_dims = [0,1,2], ard_num_dims = 3))
        return gpytorch.kernels.AdditiveKernel(k1, k2)

def get_mean(mean = 0, n_features = 0):
    """
    Returns the mean function according to the index given as an argument
    """
    if mean == 0:
        return gpytorch.means.ConstantMean()
    elif mean == 1:
        return gpytorch.means.LinearMean(n_features)
    
class GPModel(ApproximateGP):
    """
    Main ApproximateGP class. Uses variational inference to approximate and optimize the loss function.
    """
    def __init__(self, inducing_points, mean, n_features, kernel, ard):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = get_mean(mean, n_features = inducing_points.size(1))
        self.covar_module = get_kernel(kernel, ard)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ------------------------------------------------------
# helper functions
def create_loaders(t_tuple):
    """
    Creates dataloaders to load in minibatches during training and testing
    """
    x, y = t_tuple
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    return loader

def m_s_adjust(x):
    """
    Adjusts for mean and std for input/output normalization. Used in the next function.
    
    || INPUT ||
    x - dataset to standardize 
    
    || OUTPUT ||
    x - transformed dataset
    mx - mean of original dataset
    sx - std of original dataset
    """
    x = x.clone()
    mx = x.mean(0, keepdim=True)
    sx = x.std(0, unbiased=False, keepdim=True)
    x -= mx
    x /= sx
    return x, mx, sx

def normalize(train_tuple, test_tuple = None):
    """
    Normalize the train and/or test datasets
    
    || INPUT ||
    train_tuple - (train features, train output variable) 
    test_tuple - (test features, test output variable) 
    
    || OUTPUT ||
    outputs transformed train and test tuples (tuples as defined above) along with learnt mean and std. deviations
    """
    train_x, train_y = train_tuple
    train_x, train_y  = train_x.clone(), train_y.clone() 
    
    train_x, mx, sx = m_s_adjust(train_x)
    train_y, my, sy = m_s_adjust(train_y)
    train_tuple = (train_x, train_y)
    
    if test_tuple:
        test_x, test_y = test_tuple
        test_x, test_y  = test_x.clone(), test_y.clone() 
        test_x -= mx
        test_x /= sx
        test_y -= my
        test_y /= sy
        test_tuple = (test_x, test_y)
    return train_tuple, test_tuple, (mx, sx), (my, sy)

def keep_best(prev_train, prev_test, train_rmse, test_rmse):
    """
    This function helps us keep the best model learnt till now. It checks how current train and test rmse are compared to 
    previous iterations train and test rmse. Returns True if both are lower than previous.  
    """
    if train_rmse <= prev_train:
        if test_rmse <= prev_test:
            result = True
        else:
            result = False
    else:
        result = False
    return result
    
def train_gp(model, train_tuple, loader, optimizer, mll, likelihood= None, cuda = False):
    """
    Trains the Gaussian Process model
    """
    if likelihood == None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model.train()
    likelihood.train()
    minibatch_iter = tqdm.tqdm_notebook(loader, desc="Minibatch", leave=False)
    mean_loss = 0
    count = 0 
    for x_batch, y_batch in minibatch_iter:
        optimizer.zero_grad()
        output = model(x_batch)
        loss = -mll(output, y_batch)
        minibatch_iter.set_postfix(loss=loss.item())
        mean_loss += loss.item()
        count+=1
        loss.backward()
        optimizer.step()
        if cuda:
            torch.cuda.empty_cache()
    
    print("Mean Loss:", mean_loss/count)
    return model, optimizer, mll

def train_AppxGpr(train_tuple, y_ms, mean=0, kernel=0, ard = True, 
                  num_epochs = 100, ind_pts = 500, cuda = False):
    train_x, train_y = train_tuple
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    # Below selects the inducing points in such a way that they are spread
    # equally throughout the dataset
    inducing_idx = np.linspace(0, train_x.shape[0]-1, ind_pts) 
    inducing_points = train_x[inducing_idx, :]
    model = GPModel(inducing_points, mean = mean, n_features = 0, \
                    kernel = kernel, ard = ard)
       
    # transfer everything to GPU if needed
    if cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
     
    # form tuples again as depending on "cuda" they might have been transferred to GPU
    train_tuple = (train_x, train_y)
    
    # form loaders that allow us to train the data using batches
    train_loader = create_loaders(train_tuple)
  
    optimizer = torch.optim.Adam([{'params': model.parameters()}, \
                                  {'params': likelihood.parameters()}], \
                                  lr=0.05)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, \
                                        num_data=train_y.size(0))    
    
    # main chunk of code. Goes through multiple epochs 
    epochs_iter = tqdm.tqdm_notebook(range(num_epochs), desc="Epoch")
    scheduler = StepLR(optimizer, step_size=10, gamma=0.3) 
    for i in epochs_iter:
        print("Epoch:", i)        
        # training the model
        model, optimizer, mll = train_gp(model, (train_x, train_y), \
                                         train_loader, optimizer, mll, \
                                         likelihood, cuda)
        scheduler.step()
            
        print("Epoch:", i+1)
        if i%10 == 0:
            train_rmse, train_means, train_lower, train_upper = evaluate(model, \
                                                                     train_loader,\
                                                                     train_y, \
                                                                     likelihood, \
                                                                     y_ms, \
                                                                     cuda)
            print("Train RMSE:", train_rmse)
            
    return model, likelihood
        
    
def fit_model(data_tuple, num_epochs, ind_pts = 100, cuda = False, normalizer = False, kernel = 0, mean = 0, ard = None):
    """
    Fits the model. Uses all the above functions to fit the model in a customized way. 
    """
    # forming needed tuples
    train_x, train_y, test_x, test_y = data_tuple      
    train_tuple = train_x, train_y
    test_tuple = test_x, test_y
    
    # normalizing
    if normalizer:
        train_tuple, test_tuple, x_ms, y_ms = normalize(train_tuple, test_tuple)
        train_x, train_y = train_tuple
        test_x, test_y = test_tuple
    else:
        x_ms = (0,1)
        y_ms = (0, 1)
     
    ms = (x_ms, y_ms)
        
    # gaussian likelihood is part of our model definition and tells us about the pure error term
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    # model and inducing points
    inducing_idx = np.linspace(0, train_x.shape[0]-1, ind_pts) # selects the inducing points in such a way that they are spread
                                                                # equally throughout the dataset
    inducing_points = train_x[inducing_idx, :]
    model = GPModel(inducing_points, mean = mean, n_features = 0, kernel = kernel, ard = ard)
       
    # transfer everything to GPU if needed
    if cuda:
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        test_x = test_x.cuda()
        test_y = test_y.cuda()
     
    # form tuples again as depending on "cuda" they might have been transferred to GPU
    train_tuple = (train_x, train_y)
    test_tuple = (test_x, test_y)
    
    # form loaders that allow us to train the data using batches. 
    train_loader = create_loaders(train_tuple)
    test_loader = create_loaders(test_tuple)
  
    # loss and optimizer both have changed a little
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=0.05)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))    
    
    # main chunk of code. Goes through multiple epochs 
    epochs_iter = tqdm.tqdm_notebook(range(num_epochs), desc="Epoch")
    prev_train = np.inf # these are to aid us in the selection of the best model
    prev_test = np.inf
    best_model = None
    best_likelihood = None
    scheduler = StepLR(optimizer, step_size=5, gamma=0.3) # this decreases the learning rate with epochs
    for i in epochs_iter:
        print("Epoch:", i)
        
        # training the model
        model, optimizer, mll = train_gp(model, (train_x, train_y), train_loader, optimizer, mll, likelihood,cuda)
        
        # evaluating on train and validation/test set
        train_rmse, train_means, train_lower, train_upper = evaluate(model, train_loader, train_y, likelihood, y_ms, cuda)
        test_rmse, test_means, test_lower, test_upper = evaluate(model, test_loader, test_y, likelihood, y_ms, cuda)
        print('Train RMSE: {}'.format(train_rmse.item()))
        print('Test RMSE: {}'.format(test_rmse.item()))
        
        # saving best model
        model_best = keep_best(prev_train, prev_test, train_rmse, test_rmse)
        if model_best == True:
            best_model = copy.deepcopy(model)
            best_likelihood = copy.deepcopy(likelihood)
        prev_train = train_rmse
        prev_test = test_rmse
        scheduler.step()

    return (train_means, train_lower, train_upper), (test_means, test_lower, test_upper), (best_model, best_likelihood),             (model, likelihood), ms, train_rmse

def evaluate(model, loader, y, likelihood, y_ms, cuda):
    """
    Evaluates the model on the given loader and y using rmse
    """
    model.eval()
    likelihood.eval()
    
    means = torch.tensor([0.])
    lower_conf = torch.tensor([0.])
    upper_conf = torch.tensor([0.])

    with torch.no_grad():
        for x_batch, y_batch in loader:
            preds = model(x_batch)
            means = torch.cat([means, preds.mean.cpu()])
            lower, upper = preds.confidence_region()
            lower, upper = lower.cpu(), upper.cpu()
            lower_conf = torch.cat([lower_conf, lower])
            upper_conf = torch.cat([upper_conf, upper]) 
    
    means = means[1:]
    if cuda:
        y = y.cpu()
    lower_conf = lower_conf[1:]
    upper_conf = upper_conf[1:]
    unnormalized_means = means*y_ms[1] + y_ms[0]
    lower_conf = lower_conf*y_ms[1] + y_ms[0]
    upper_conf = upper_conf*y_ms[1] + y_ms[0]
    
    prediction_error = means - y
    rmse = torch.sqrt(torch.mean(torch.square(prediction_error)))
    unnormalized_rmse = y_ms[1]*rmse
    
    return unnormalized_rmse, unnormalized_means, lower_conf, upper_conf

# def eval_fresh(model, likelihood, t_tuple, ms, normalizer, cuda):
#     x, y = t_tuple
#     x_ms, y_ms = ms
#     x, y = x.clone(), y.clone()
#     if normalizer:
#         x -= x_ms[0]
#         x /= x_ms[1]
#         y -= y_ms[0]
#         y /= y_ms[1]
        
#     if cuda:
#         x = x.cuda()
#         y = y.cuda()
#         model = model.cuda()
#         likelihood = likelihood.cuda()
        
#     loader = create_loaders((x,y))
  
#     return evaluate(model, loader, y, likelihood, y_ms, cuda)

def reconstruct(sub_df, preds):
    """
    forms a dataset using the orgiginal and predicted data
    """
    cols = sub_df.columns.tolist()
    cols.extend(["means", "lc", "uc"])
    means, lower, upper = preds
    means, lower, upper = means.detach().numpy().reshape(-1,1), lower.detach().numpy().reshape(-1,1), upper.detach().numpy().reshape(-1,1)
    sub = pd.DataFrame(np.hstack([sub_df, means, lower, upper]), columns = cols)
    sub = sub.sort_values("dateTime")
    return sub
#---------------------------------------------------------------------
### functions for hyperparam printing 

# def get_hp_dict(cons_hp):
    
#     hp_dict = {}
#     for i in cons_hp:
#         if ("base_kernel" in i[0]) or ("kernels" in i[0]):
#             pass
#         else:
#             curr_item = i[1]
#             if curr_item.shape==():
#                 curr_item_list = [i[2].transform(curr_item)]
#             else:
#                 curr_item_list = []
#                 for item in curr_item:
#                     curr_item_list.append(i[2].transform(item)) 
#             hp_dict[i[0][4:]] = curr_item_list[0].detach().numpy()
#     return hp_dict

# def get_children(current):
#     children = []
#     for i in current.named_children():
#         if "kernel" not in i[0]:
#             pass
#         else:
#             children.append(i)
#     return children
                
# def hyperparams_simple(model):
#     """
#     Print the models hyperparameters for hyperparameter exploration. Only works for kernels not using additive or product 
#     kernels
#     """
#     kern_dict = {}
#     current = model.covar_module
#     count = 0
#     while True:
#         name = str(count) + "_" + str(type(current))[25:-2]
#         children = get_children(current)
#         cons_hp = [i for i in current.named_parameters_and_constraints()]
#         kern_dict[name] = get_hp_dict(cons_hp)
#         if children == []:
#             break
#         else:
#             current = children[0][1]
#         count += 1
#     return pd.DataFrame(kern_dict).T
    
df = pd.read_csv("../data/2020-11-3_all.csv", index_col=0, parse_dates=["dateTime"])
df["datetime"] = df.dateTime.apply(lambda x: x.timestamp())

# create data tuple. You can change this frac below as per your requirements.
sub_train = df.sample(frac = 0.8, random_state = 1)
sub_test = df.drop(sub_train.index, axis = 0)

print("Train and Test rows:", sub_train.shape[0], ",",sub_test.shape[0])


all_cols = ["lat", "long", "datetime", "pm10"] 
x_cols = ["lat", "long", "datetime"]
y_cols = ["pm10"]

x_train, y_train = torch.Tensor(sub_train[x_cols].values), torch.Tensor(sub_train[y_cols].values.flatten())
x_test, y_test = torch.Tensor(sub_test[x_cols].values), torch.Tensor(sub_test[y_cols].values.flatten())

data_tuple = (x_train, y_train, x_test, y_test)

train_preds, test_preds, best_model, curr_model, ms, train_rmse  = fit_model(data_tuple, 15, 2500, normalizer = True, mean=0, 
                                                                              kernel = 6, cuda= True)

