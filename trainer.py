import os
import time
import json
import torch
import numpy as np
import argparse
import subprocess
from torch.utils.data import DataLoader

import losses
import bayesian_deepssm
import datasets

torch.cuda.manual_seed_all(0)

DATA_DIR = 'data/'

'''
Train helper
	Initilaizes all weights using initialization function specified by initf
'''
def weight_init(module, initf):
	def foo(m):
		classname = m.__class__.__name__.lower()
		if isinstance(m, module):
			initf(m.weight)
	return foo

'''
Train helper
	prints and logs values during training
'''
def log_print(logger, values):
	# write to csv
	if not isinstance(values[0], str):
		values = ['%.5f' % i for i in values]
	string_values = [str(i) for i in values]
	log_string = ','.join(string_values)
	logger.write(log_string + '\n')
	# print
	for i in range(len(string_values)):
		while len(string_values[i]) < 15:
			string_values[i] += ' '
	print(' '.join(string_values))

'''
Train helper
	Learning rate scheduler
'''
def set_scheduler(opt, sched_params):
	if sched_params["type"] == "Step":
		step_size = sched_params['parameters']['step_size']
		gamma = sched_params['parameters']['gamma']
		scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.99)
	elif sched_params["type"] == "CosineAnnealing":
		T_max = sched_params["parameters"]["T_max"]
		eta_min = sched_params["parameters"]["eta_min"]
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=T_max, eta_min=eta_min)
	elif sched_params["type"] == "Exponential":
		gamma = sched_params['parameters']['gamma']
		scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
	else:
		print("Error: Learning rate scheduler not recognized or implemented.")
	return scheduler

'''
Network training method
	defines, initializes, and trains the models
	logs training and validation errors
	saves the model
'''
def train(config_file):
	### Initializations

	# Get parameter dictionary
	with open(config_file) as json_file: 
		parameters = json.load(json_file)

	# Get data loaders
	print("Loading data...")
	train_dataset = datasets.get_dataset(parameters['dataset'], 'train', parameters['train_size'], parameters['trainer']['histogram_aug'], parameters['trainer']['noise_aug'])
	train_loader = DataLoader(train_dataset, batch_size=parameters['trainer']['batch_size'], shuffle=True)
	print("Train data size:", len(train_dataset))
	val_dataset = datasets.get_dataset(parameters['dataset'], 'val')
	val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
	print("Val data size:", len(val_dataset))

	# Get loss
	loss_func = getattr(losses, parameters["loss"]["function"])
	loss_params = parameters["loss"]["params"]

	# Define model dir
	aug = ''
	if parameters['trainer']['histogram_aug']:
		aug += '__histogram'
	if parameters['trainer']['noise_aug']:
		aug += '__noise'
	model_dir = parameters['out_dir'] + parameters['dataset'] + '/' + parameters['model_name'] + '__' + parameters['loss']['function'] + aug + '/'
	print("Saving to", model_dir)
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)

	# Initialize logger
	logger = open(model_dir + "train_log.csv", "w+")
	log_header = ["Epoch", "LR", "train_loss", "train_y_mse", "val_y_mse", "train_cd", "val_cd", "z_logsig_mean", "y_logsig_mean", "Sec"]

	# Define the model
	print("Defining model...")
	model = bayesian_deepssm.BayesianDeepSSMNet(config_file, train_dataset)
	device = model.device
	model.to(device)
	# Model initialization  - xavier
	model.apply(weight_init(module=torch.nn.Conv3d, initf=torch.nn.init.xavier_normal_))	
	model.apply(weight_init(module=torch.nn.Linear, initf=torch.nn.init.xavier_normal_))

	# # Initialize log_var weights to be very small
	# if parameters['encoder']['stochastic']:
	# 	# Initialize small random log_var weights
	# 	torch.nn.init.normal_(model.encoder.pred_z_dist.weight, mean=0.0, std=1e-6)
	# 	torch.nn.init.normal_(model.encoder.pred_z_dist.bias, mean=0.0, std=1e-6)

	# If initializing using pretrained model
	if not parameters['initialize_model'] is None:
		print("Loading previously trained model", parameters['initialize_model'])
		logger.write("Loading previously trained model: "+ parameters['initialize_model'])
		model_dict = model.state_dict()
		intial_model_dict = torch.load(parameters['initialize_model'], map_location=model.device)
		# initialze weights that are common between new and pretrained model
		pretrained_dict = {k: v for k, v in intial_model_dict.items() if k in model_dict}
		# pretrained_dict = {k: v for k, v in intial_model_dict.items() if k in model_dict and 'encoder.ConvolutionalBackbone.fc_blocks' not in k}
		model_dict.update(pretrained_dict) 
		model.load_state_dict(model_dict)
		print(f'Initializing weights for: {pretrained_dict.keys()}. ')
		not_pretrained_dict = {k: v for k, v in model_dict.items() if k not in pretrained_dict}
		print(f'No pretrained initialization for: {not_pretrained_dict.keys()}. ')

	# Initialize linear decoder weights with PCA if requested
	if parameters['decoder']['pca_intialized']:
		print("Initializing decoder with PCA")
		logger.write("Initializing decoder with PCA\n")
		num_pca = parameters['num_latent_dim']
		num_corr = train_loader.dataset.corrs[0].shape[0]
		orig_mean = np.loadtxt(DATA_DIR + parameters['dataset'] + '/PCA_Particle_Info/mean.particles')
		orig_pc = np.zeros([num_pca, num_corr*3])
		for i in range(num_pca):
			temp = np.loadtxt(DATA_DIR + parameters['dataset'] + '/PCA_Particle_Info/pcamode' + str(i) + '.particles')
			orig_pc[i, :] = temp.flatten()
		bias = torch.from_numpy(orig_mean.flatten()).to(device) # load the mean here
		weight = torch.from_numpy(orig_pc.T).to(device) # load the PCA vectors here
		model.decoder.pred_y_mean.bias.data.copy_(bias)
		model.decoder.pred_y_mean.weight.data.copy_(weight)
		
	# Fix the decoder weight if requested by setting the gradient to zero
	if parameters['decoder']['fixed']:
		print("Fixing decoder weights")
		logger.write("Fixing decoder weights\n")
		for param in model.decoder.pred_y_mean.parameters():
			param.requires_grad = False
	
	# Define the optimizer
	opt = torch.optim.Adam(model.parameters(), parameters['trainer']['learning_rate'], weight_decay=parameters['trainer']['weight_decay'])
	opt.zero_grad()
	
	# Define the learning rate scheduler
	if parameters['trainer']['decay_lr']['enabled']:
		scheduler = set_scheduler(opt, parameters['trainer']['decay_lr'])
	print("Done.")

	
	# Intialize training variables
	t0 = time.time()
	best_val_error = np.Inf
	patience_count = 0
	
	### Train
	print("Beginning training on device = " + device)
	log_print(logger, log_header)
	torch.cuda.empty_cache()
	
	# Loop over epochs
	num_samples = parameters['trainer']['num_samples']
	model.train()
	for e in range(1, parameters['trainer']['epochs'] + 1):
		# torch.cuda.empty_cache()
		train_losses = []
		loss_params['epoch'] = e

		if "burnin" in parameters["loss"]["function"]:
			# burn in sampling
			if e < parameters['loss']['params']['initiate_stochastic']/4:
				num_samples = 0
			else:
				num_samples = parameters['trainer']['num_samples']

		# burn in dropout
		enable_dropout = True
		if parameters["dropout"]["type"] == "concrete":
			if e < parameters['dropout']['params']['start_epoch']:
				for i in range(5):
					model.encoder.ConvolutionalBackbone.conv_dropouts[i].p_logit.requires_grad = False
				for i in range(2):
					model.encoder.ConvolutionalBackbone.fc_dropouts[i].p_logit.requires_grad = False
				if parameters['decoder']["stochastic"]:
					for i in range(3):
						model.decoder.fc_dropouts[i].p_logit.requires_grad = False
				# enable_dropout = False
			else:
				for param in model.parameters():
				    param.requires_grad = True

		z_log_sigma_means = []
		y_log_sigma_means = []
		for img, pc, mdl, z, name in train_loader:
			opt.zero_grad()
			batch_size = img.shape[0]
			img = img.to(device)
			pc = pc.to(device)
			mdl = mdl.to(device).flatten(start_dim=1)
			if parameters['batch_ensemble']['enabled']:
				img = torch.cat([img for i in range(parameters['batch_ensemble']['num_models'])], dim=0)
				pc = torch.cat([pc for i in range(parameters['batch_ensemble']['num_models'])], dim=0)
				mdl = torch.cat([mdl for i in range(parameters['batch_ensemble']['num_models'])], dim=0)
			pred_z, pred_mdl, regularization = model(img, num_samples=num_samples, use_dropout=enable_dropout)			
			loss_ = loss_func(pred_z, pred_mdl, z, mdl, pc, loss_params)

			if parameters["dropout"]["type"]=="concrete":
				loss = loss_ + regularization
			else:
				loss = loss_

			if parameters['batch_ensemble']['enabled']:
				pred_z[0] = torch.mean(pred_z[0].reshape((parameters['batch_ensemble']['num_models'], batch_size,) + pred_z[0].shape[1:]), 0)
				pred_z[1] = torch.mean(pred_z[1].reshape((parameters['batch_ensemble']['num_models'], batch_size,) + pred_z[1].shape[1:]), 0)
				pred_mdl[0] = torch.mean(pred_mdl[0].reshape((parameters['batch_ensemble']['num_models'], batch_size,) + pred_mdl[0].shape[1:]), 0)
				pred_mdl[1] = torch.mean(pred_mdl[1].reshape((parameters['batch_ensemble']['num_models'], batch_size,) + pred_mdl[1].shape[1:]), 0)

			z_log_sigma_means.append(torch.mean(pred_z[1]).item())
			y_log_sigma_means.append(torch.mean(pred_mdl[1]).item())
	
			if not torch.isnan(loss):
				loss.backward()
				if parameters['trainer']['gradient_clipping']:
					torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
				opt.step()
				train_losses.append(loss.item())
			else:
				print("z_mean", torch.sum(pred_z[0]))
				print("z_sigma", torch.sum(pred_z[1]))
				print("y_mean", torch.sum(pred_mdl[0]))
				print("y_sigma", torch.sum(pred_mdl[1]))				
				print("loss", loss_)
				print("reg", regularization)
				print("combined", loss)
				print(("Error:Loss is NAN"))
				parameters['NAN'] = True
				patience_count = parameters['trainer']['early_stop']['patience']
				break
		train_loss = np.mean(train_losses)
		z_log_sigma_mean = np.mean(z_log_sigma_means)
		y_log_sigma_mean = np.mean(y_log_sigma_means)
		# Test
		if parameters['batch_ensemble']['enabled']:
			train_corr_mse, train_cd = be_test_mse(model, train_loader, parameters['batch_ensemble']['num_models'])
			val_corr_mse, val_cd = be_test_mse(model, val_loader, parameters['batch_ensemble']['num_models'])
		else:
			train_corr_mse, train_cd = test_mse(model, train_loader, parameters['encoder']['offset'])
			val_corr_mse, val_cd = test_mse(model, val_loader, parameters['encoder']['offset'])
		log_print(logger, [e, opt.param_groups[0]['lr'], train_loss, train_corr_mse, val_corr_mse, train_cd, val_cd, z_log_sigma_mean, y_log_sigma_mean, time.time()-t0])
		
		# Print dropout probs for concrete
		if parameters["dropout"]["type"]=="concrete": # and e%10==0:
			model_dict = model.state_dict()
			drop_keys = [key for key in model_dict.keys() if "p_logit" in key]
			Ps = torch.empty(len(drop_keys))
			for i in range(len(drop_keys)):
				Ps[i] = torch.sigmoid(model_dict[drop_keys[i]])
			print("Dropout probs: ", Ps.numpy())
			logger.write("Dropout probs: " + str(Ps.numpy()) + '\n')

		# Save is requested
		if parameters['trainer']['save_iter_freq']:
			if e%parameters['trainer']['save_iter_freq'] == 0:
				torch.save(model.state_dict(), os.path.join(model_dir, 'model_epoch_'+str(e)+'.torch'))
		if e >= parameters['trainer']['early_stop']['start_epoch']:
			if 'chamfer' in parameters['loss']['function']:
				val_error = val_cd
			else:
				val_error = val_corr_mse
			if val_error < best_val_error:
				best_val_error = val_error
				best_epoch = e
				torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.torch'))
				print("Best error:",str(best_val_error)[:5],"Saving.")
				patience_count = 0
			# Check early stoppping criteria
			else:
				patience_count += 1
				if parameters['trainer']['early_stop']['enabled']:
					if patience_count >= parameters['trainer']['early_stop']['patience']:
						break
		
		# Check learning rate decay criteria
		if parameters['trainer']['decay_lr']['enabled']:
			scheduler.step()

		# print(torch.sum(model.encoder.pred_z_log_var.weight).item())
		t0 = time.time()
	
	# Save final model
	logger.close()
	torch.save(model.state_dict(), os.path.join(model_dir, 'final_model.torch'))
	
	# Write epoch at which best model was saved
	parameters['best_model_epochs'] = best_epoch
	out_config = model_dir+os.path.basename(config_file)
	with open(out_config, "w") as json_file:
		json.dump(parameters, json_file, indent=2) 
	print("Training complete, model saved. Best model after epoch " + str(best_epoch) + '\n')
	subprocess.call(['python', 'eval.py', '-c', out_config, '-n', '100'])


'''
Test on given loader 
'''
def test_mse(model, loader, offset=False):
	device = model.device
	model.eval()
	with torch.no_grad():
		corr_mses = []
		cds = []
		for img, pc, mdl, z, name in loader:
			img = img.to(device)
			pc = pc.to(device)
			mdl = mdl.to(device).flatten(start_dim=1)
			pred_z, pred_y, _ = model(img, num_samples=0, use_dropout=False)
			if offset:
				pred = pred_z[2] + pred_y[0]
			else:
				pred = pred_y[0]
			corr_mses.append(torch.mean((pred- mdl)**2).item())
			cds.append(losses.CD(pred,pc).detach().cpu().numpy())
		corr_mse = np.mean(corr_mses)
		cd = np.mean(cds)
		return corr_mse, cd

'''
Test on given loader with batch ensemble
'''
def be_test_mse(model, loader, num_models):
	device = model.device
	model.eval()
	with torch.no_grad():
		corr_mses = []
		cds = []
		for img, pc, mdl, z, name in loader:
			batch_size = img.shape[0]
			img = img.to(device)
			pc = pc.to(device)
			img = torch.cat([img for i in range(num_models)], dim=0)
			mdl = mdl.to(device).flatten(start_dim=1)
			pred_z, pred_y, _ = model(img, num_samples=0, use_dropout=False)
			pred = torch.mean(pred_y[0].reshape((num_models, batch_size,) + pred_y[0].shape[1:]), 0)
			corr_mses.append(torch.mean((pred- mdl)**2).item())
			cds.append(losses.CD(pred,pc).detach().cpu().numpy())
		corr_mse = np.mean(corr_mses)
		cd = np.mean(cds)
		return corr_mse, cd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    train(arg.config)