import torch
import torch.distributions as dist
import math
import numpy as np
import pytorch3d
from pytorch3d import loss

from torch.distributions.multivariate_normal import MultivariateNormal
epsilon=1e-6 

def pca_mse(pred_z, pred_y, true_z, true_y, point_cloud, params):
	return MSE(pred_z[0], true_z)

def particle_mse(pred_z, pred_y, true_z, true_y, point_cloud, params):
	return MSE(pred_y[0], true_y)

def pca_nll(pred_z, pred_y, true_z, true_y, point_cloud, params):
	return NLL(pred_z[0], pred_z[1], true_z)

def pca_nll_burnin(pred_z, pred_y, true_z, true_y, point_cloud, params):
	z_mu = pred_z[0]
	z_log_sigma = pred_z[1]
	epoch = params["epoch"]
	init = params['initiate_stochastic']
	comp = params['complete_stochastic']
	y_mse = MSE(pred_y[0], true_y)	
	# Deterministic phase
	if epoch <= init:
		alpha = 1 - epsilon
	# Introduce stochastic
	else:
		alpha = min(1, ((epoch - init)/(comp - init)))
	z_nll = NLL(z_mu, z_log_sigma, true_z)
	loss = (1-alpha)*y_mse + alpha*z_nll
	return loss

def vib(pred_z, pred_y, true_z, true_y, point_cloud, params):
	beta = params['beta']
	y_nll = NLL(pred_y[0], pred_y[1].to(pred_y[0].device), true_y)
	z_kld = KLD(pred_z[0], pred_z[1])
	loss = y_nll + beta*z_kld
	return loss

def vib_burnin(pred_z, pred_y, true_z, true_y, point_cloud, params):
	epoch = params["epoch"]
	beta = params['beta']
	init = params['initiate_stochastic']
	comp =params['complete_stochastic']
	y_mse = MSE(pred_y[0], true_y)
	y_nll = NLL(pred_y[0], pred_y[1].to(pred_y[0].device), true_y)
	z_kld = KLD(pred_z[0], pred_z[1])
	# Deterministic phase
	if epoch <= init:
		loss = y_mse
	# Introduce stochastic
	else:
		alpha = min(1, ((epoch - init)/(comp - init)))
		loss = (1-alpha)*y_mse + alpha*y_nll + alpha*beta*z_kld
	return loss

def vib_chamfer(pred_z, pred_y, true_z, true_y, point_cloud, params):
	beta = params['beta']
	cd_reg_weight = params['cd_reg_weight']
	mu, log_sigma = pred_y
	if len(mu.shape) == 2:
		mu = mu.reshape((mu.shape[0], mu.shape[1]//3, 3))
	log_sigma = log_sigma + epsilon
	if len(log_sigma.shape) == 2:
		log_sigma = log_sigma.reshape((mu.shape[0], log_sigma.shape[1]//3, 3))
	log_sigma = log_sigma.sum(axis=-1).to(mu.device)
	y_cd, _ = pytorch3d.loss.chamfer_distance(mu, point_cloud, norm=2, batch_reduction=None, point_reduction=None, single_directional=True)
	y_cd_nll = torch.mean(0.5 * (log_sigma + y_cd / torch.exp(log_sigma))) # + 0.5 * math.log(2 * math.pi)
	y_cd = y_cd.mean()
	cd_point_cloud, _ = pytorch3d.loss.chamfer_distance(point_cloud.to(mu.device), mu, norm=2, single_directional=True)
	z_kld = KLD(pred_z[0], pred_z[1])
	loss = y_cd_nll + beta*z_kld +cd_reg_weight*cd_point_cloud
	return loss

def vib_chamfer_burnin(pred_z, pred_y, true_z, true_y, point_cloud, params):
	epoch = params["epoch"]
	beta = params['beta']
	init = params['initiate_stochastic']
	comp = params['complete_stochastic']
	cd_reg_weight = params['cd_reg_weight']
	mu, log_sigma = pred_y
	if len(mu.shape) == 2:
		mu = mu.reshape((mu.shape[0], mu.shape[1]//3, 3))
	log_sigma = log_sigma + epsilon
	if len(log_sigma.shape) == 2:
		log_sigma = log_sigma.reshape((mu.shape[0], log_sigma.shape[1]//3, 3))
	log_sigma = log_sigma.sum(axis=-1).to(mu.device)
	y_cd, _ = pytorch3d.loss.chamfer_distance(mu, point_cloud.to(mu.device), norm=2, batch_reduction=None, point_reduction=None, single_directional=True)
	y_cd_nll = torch.mean(0.5 * (log_sigma + y_cd / torch.exp(log_sigma))) # + 0.5 * math.log(2 * math.pi)
	y_cd = y_cd.mean()
	cd_point_cloud, _ = pytorch3d.loss.chamfer_distance(point_cloud.to(mu.device), mu, norm=2, single_directional=True)
	z_kld = KLD(pred_z[0], pred_z[1])
	# Deterministic phase
	if epoch <= init:
		alpha = 1 - epsilon
	# Introduce stochastic
	else:
		alpha = min(1, ((epoch - init)/(comp - init)))
	loss = (1-alpha)*y_cd + (alpha*y_cd_nll + alpha*beta*z_kld) + cd_reg_weight*cd_point_cloud
	return loss

def chamfer(pred_z, pred_y, true_z, true_y, point_cloud, params):
	mu, log_sigma = pred_y
	if len(mu.shape) == 2:
		mu = mu.reshape((mu.shape[0], mu.shape[1]//3, 3))
	loss, _ = pytorch3d.loss.chamfer_distance(mu, point_cloud, norm=2)
	return loss

####### Helper functions

def MSE(predicted, true):
	return torch.mean((predicted - true.to(predicted.device))**2)

def NLL(mu, log_sigma, ground_truth):
	log_sigma = log_sigma + epsilon
	nll_loss = 0.5 * (log_sigma + (mu - ground_truth.to(mu.device))**2 / torch.exp(log_sigma)) # + 0.5 * math.log(2 * math.pi)
	return torch.mean(nll_loss)

def CD(predicted, true):
	if len(predicted.shape) == 2:
		predicted = predicted.reshape((predicted.shape[0], predicted.shape[1]//3, 3))
	cd_l2, _ = pytorch3d.loss.chamfer_distance(predicted, true, norm=2, point_reduction='mean')
	return cd_l2

def KLD(mu, log_sigma):
	log_sigma = log_sigma + epsilon
	kld = -0.5 * (1 + log_sigma - mu.pow(2) - (log_sigma).exp()) 
	return torch.mean(kld)
