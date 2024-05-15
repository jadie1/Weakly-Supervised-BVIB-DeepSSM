# Jadie Adams
import os
import json
import numpy as np
from numpy import matlib
import torch
from torch.utils.data import DataLoader
import re
from scipy.stats import pearsonr
import json
import time
import argparse

import datasets
import losses
import bayesian_deepssm

DATA_DIR = 'data/'


def test(config_file, loader_name, samples=0):
	# Get params
	with open(config_file) as json_file: 
		parameters = json.load(json_file)
	aug = ''
	if parameters['trainer']['histogram_aug']:
		aug += '__histogram'
	if parameters['trainer']['noise_aug']:
		aug += '__noise'
	model_dir = parameters['out_dir'] + parameters['dataset'] + '/' + parameters['model_name'] + '__' + parameters['loss']['function'] + aug + '/'
	pred_dir = model_dir + loader_name+'/'
	if not os.path.exists(pred_dir):
		os.makedirs(pred_dir)
	for folder in ['particles/', 'aleatoric/', 'epistemic/', 'point_clouds/', 'latent_mean/', 'latent_logvar/']:
		if not os.path.exists(pred_dir + folder):
			os.makedirs(pred_dir + folder)
	if parameters["use_best_model"]:
		model_path = model_dir + 'best_model.torch'
	else:
		model_path = model_dir + 'final_model.torch'
	num_latent = parameters["num_latent_dim"]
	# initalizations
	dataset = datasets.get_dataset(parameters['dataset'], loader_name)
	loader = DataLoader(dataset, batch_size=1, shuffle=False)

	print("Loading model " + model_path + "...")
	print("Epochs", parameters["best_model_epochs"])
	model = bayesian_deepssm.BayesianDeepSSMNet(config_file, dataset)
	model.load_state_dict(torch.load(model_path, map_location=model.device))
	device = model.device
	model.to(device)
	model.eval()
	with torch.no_grad():
		enable_dropout(model)
		# enable_mixup(model)
		# if parameters['batch_ensemble']['enabled'] and parameters['dropout']['type']=="concrete":
		# 	be_conc_test_loader(model, name, loader_dir, pred_dir, samples, parameters)
		if parameters['batch_ensemble']['enabled']:
			be_test_loader(model, loader_name, loader, pred_dir, samples, parameters)
		else:
			test_loader(model, loader_name, loader, pred_dir, samples, parameters)

def enable_dropout(model):
	count = 0
	for m in model.modules():
		if 'Dropout' in m.__class__.__name__:
			count += 1
			m.train()
	print("Turned on", count, "dropout layers.")
	model_dict = model.state_dict()
	drop_keys = [key for key in model_dict.keys() if "p_logit" in key]
	if drop_keys:
		Ps = torch.empty(10)
		for i in range(count):
			Ps[i] = torch.sigmoid(model_dict[drop_keys[i]])
		print("Dropout probs: ", Ps.numpy())

def enable_mixup(model):
	model_dict = model.state_dict()
	be_keys = [key for key in model_dict.keys() if "alpha" in key]
	for i in range(5):
		model.encoder.ConvolutionalBackbone.conv_blocks[i].conv_block[0].mixup = True
	model.encoder.ConvolutionalBackbone.fc_blocks[0].fc_block[1].mixup = True
	model.encoder.ConvolutionalBackbone.fc_blocks[1].fc_block[0].mixup = True
	model.encoder.pred_z_dist.mixup = True
	for i in range(3):
		model.decoder.fc_blocks[i].fc_block[0].mixup = True
	model.decoder.pred_y.mixup=True

def test_loader(model, name, loader, pred_dir, samples, parameters):
	det_CDs, vib_CDs = [], []
	det_MSES, vib_MSES = [], []
	ale_uncs, epi_uncs = [], []
	img_out, shape_out = [], []
	individual_results = {}
	with open(DATA_DIR + parameters['dataset'] + "/data_info.json") as json_file:
		data_info = json.load(json_file) 
	for img, pc, corr, z, nm in loader:
		nm = nm[0]
		num_particles = corr.shape[1]
		corr = corr.flatten(start_dim=1)
		# Deterministic - mean Z as sample
		_, det_pred_y, _ =  model(img.to(model.device), 0, use_dropout=False)
		det_MSES.append(torch.mean((det_pred_y[0] - corr.to(model.device))**2).item())
		det_CDs.append(losses.CD(det_pred_y[0], pc.to(model.device)).detach().cpu())
		# Aleatoric - Avg over z samples
		pred_z, pred_y, _ = model(img.to(model.device), samples, use_dropout=False)
		np.savetxt(pred_dir + 'particles/' + nm + ".particles", det_pred_y[0].detach().cpu().numpy().reshape((num_particles,3)))
		np.savetxt(pred_dir + 'point_clouds/' + nm + ".particles", pc[0].detach().cpu().numpy())
		np.savetxt(pred_dir + 'latent_mean/'+nm+".particles", pred_z[0].detach().cpu().numpy())
		np.savetxt(pred_dir + 'latent_logvar/'+nm+".particles", pred_z[1].detach().cpu().numpy())
		MSE = torch.mean((pred_y[0] - corr.to(model.device))**2).item()
		CD = losses.CD(pred_y[0], pc.to(model.device)).detach().cpu().item()
		vib_CDs.append(CD)
		vib_MSES.append(MSE)
		ale_unc = np.exp(pred_y[1].detach().cpu().numpy())
		ale_uncs.append(np.sum(ale_unc))
		np.savetxt(pred_dir + 'aleatoric/' + nm + ".npy", ale_unc.reshape((num_particles,3)))
		# Epistemic - Avg over dropout masks
		sampled_y_mus = []
		for i in range(samples):
			pred_z, pred_y, _ = model(img.to(model.device), 0, use_dropout=True)
			sampled_y_mus.append(pred_y[0].detach().cpu().numpy())
		epi_unc = np.var(np.array(sampled_y_mus), axis=0)
		epi_uncs.append(np.sum(epi_unc))
		np.savetxt(pred_dir + 'epistemic/' + nm + ".npy", epi_unc.reshape((num_particles,3)))
		# Outlier degrees 
		img_out.append(data_info[nm]["image"])
		shape_out.append(data_info[nm]["shape"])
		# Record individual results
		individual_results[nm] = data_info[nm]
		individual_results[nm]["MSE"] = str(MSE) 
		individual_results[nm]["CD"] = str(CD)
		individual_results[nm]["aleatoric"] = str(np.sum(ale_unc))
		individual_results[nm]["epistemic"] = str(np.sum(epi_unc))
	results = {}
	print(name + " results: ")
	print("Mean " + name + "  det CD = " +str(round(np.mean(det_CDs),4))+'+-'+str(round(np.std(det_CDs), 4)))
	results["detCD"] = {"mean":str(round(np.mean(det_CDs),4)), "std":str(round(np.std(det_CDs), 4))}
	print("Mean " + name + "  VIB CD = " +str(round(np.mean(vib_CDs),4))+'+-'+str(round(np.std(vib_CDs), 4)))
	results["vibCD"] = {"mean":str(round(np.mean(vib_CDs),4)), "std":str(round(np.std(vib_CDs), 4))}
	print("Mean " + name + "  det MSE = " +str(round(np.mean(det_MSES),4))+'+-'+str(round(np.std(det_MSES), 4)))
	results["detMSE"] = {"mean":str(round(np.mean(det_MSES),4)), "std":str(round(np.std(det_MSES), 4))}
	print("Mean " + name + "  VIB MSE = " +str(round(np.mean(vib_MSES),4))+'+-'+str(round(np.std(vib_MSES), 4)))
	results["vibMSE"] = {"mean":str(round(np.mean(vib_MSES),4)), "std":str(round(np.std(vib_MSES), 4))}

	corr, _ = pearsonr(vib_CDs, ale_uncs)
	results["ale/CD"] = str(corr)
	print("Aleatoric uncertainty / CD correlation:", corr)
	corr, _ = pearsonr(vib_CDs, epi_uncs)
	results["epi/CD"] = str(corr)
	print("Epistemic uncertainty / CD correlation:", corr)
	corr, _ = pearsonr(vib_CDs, [sum(x) for x in zip(ale_uncs, epi_uncs)])
	results["pred/CD"] = str(corr)
	print("Predictive uncertainty / CD correlation:", corr)

	corr, _ = pearsonr(vib_MSES, ale_uncs)
	results["ale/MSE"] = str(corr)
	print("Aleatoric uncertainty / MSE correlation:", corr)
	corr, _ = pearsonr(vib_MSES, epi_uncs)
	results["epi/MSE"] = str(corr)
	print("Epistemic uncertainty / MSE correlation:", corr)
	corr, _ = pearsonr(vib_MSES, [sum(x) for x in zip(ale_uncs, epi_uncs)])
	results["pred/MSE"] = str(corr)
	print("Predictive uncertainty / MSE correlation:", corr)
	corr, _ = pearsonr(ale_uncs, img_out)
	results["ale/out"] = str(corr)
	print("Aleatoric uncertainty / Image outlier correlation:", corr)
	corr, _ = pearsonr(epi_uncs, shape_out)
	print("Epsitemic uncertainty / Shape outlier correlation:", corr)
	print("Aleatoric magnitude:", round(np.mean(ale_uncs),4), "+-", np.std(ale_uncs))
	results["aleatoric"] = {"mean":str(round(np.mean(ale_uncs),4)), "std":str(round(np.std(ale_uncs), 4))}
	print("Epistmeic magnitude:", np.mean(epi_uncs), "+-", np.std(epi_uncs))
	results["epistemic"] = {"mean":str(round(np.mean(epi_uncs),4)), "std":str(round(np.std(epi_uncs), 4))}
	with open(pred_dir+name+"_overall_results.json", "w") as outfile:
		json.dump(results, outfile, indent=2)
	with open(pred_dir+name+"_individual_results.json", "w") as outfile:
		json.dump(individual_results, outfile, indent=2)

def be_test_loader(model, name, loader, pred_dir, samples, parameters):
	# print("Mixup:")
	# print(model.encoder.ConvolutionalBackbone.conv_blocks[0].conv_block[0].mixup)
	# print(model.encoder.ConvolutionalBackbone.fc_blocks[0].fc_block[1].mixup)
	# print(model.encoder.pred_z_dist.mixup)
	# print(model.decoder.fc_blocks[0].fc_block[0].mixup)
	# print(model.decoder.pred_y.mixup)
	num_models = parameters['batch_ensemble']['num_models']
	det_MSES, vib_MSES = [], []
	det_CDs, vib_CDs = [],[]
	ale_uncs, epi_uncs = [], []
	img_out, shape_out = [], []
	individual_results = {}
	with open(DATA_DIR + parameters['dataset'] + "/data_info.json") as json_file:
		data_info = json.load(json_file) 
	for img, pc, corr, z, nm in loader:
		nm = nm[0]
		num_particles = corr.shape[1]
		corr = corr.flatten(start_dim=1)
		batch_size = img.shape[0]
		img = torch.cat([img for i in range(num_models)], dim=0)
		# Deterministic z
		_, y, _ =  model(img.to(model.device), 0, use_dropout=False)
		# pred_y = torch.mean(y[0].reshape((num_models, batch_size,) + y[0].shape[1:]), axis=0)
		pred_y = y[0].reshape((num_models, batch_size,) + y[0].shape[1:])[0]
		det_MSES.append(torch.mean(((pred_y - corr.to(model.device))**2)).item())
		det_CDs.append(losses.CD(pred_y, pc.to(model.device)).detach().cpu())
		# Epistemic uncertainty
		epi_unc = torch.var(y[0], axis=0).detach().cpu().numpy()
		epi_uncs.append(np.sum(epi_unc))
		np.savetxt(pred_dir + 'epistemic/' + nm + ".npy", epi_unc.reshape((num_particles,3)))
		# Aleatoric - Avg over z samples
		_, y, _ = model(img.to(model.device), samples, use_dropout=False)
		pred_y = torch.mean(y[0].reshape((num_models, batch_size,) + y[0].shape[1:]), axis=0)
		os.makedirs(pred_dir + 'particles1/', exist_ok=True)
		np.savetxt(pred_dir + 'particles1/' + nm + ".particles", y[0][0].detach().cpu().numpy().reshape((num_particles,3)))
		os.makedirs(pred_dir + 'particles2/', exist_ok=True)
		np.savetxt(pred_dir + 'particles2/' + nm + ".particles", y[0][1].detach().cpu().numpy().reshape((num_particles,3)))
		os.makedirs(pred_dir + 'particles3/', exist_ok=True)
		np.savetxt(pred_dir + 'particles3/' + nm + ".particles", y[0][2].detach().cpu().numpy().reshape((num_particles,3)))
		os.makedirs(pred_dir + 'particles4/', exist_ok=True)
		np.savetxt(pred_dir + 'particles4/' + nm + ".particles", y[0][3].detach().cpu().numpy().reshape((num_particles,3)))
		np.savetxt(pred_dir + 'particles/' + nm + ".particles", pred_y.detach().cpu().numpy().reshape((num_particles,3)))
		np.savetxt(pred_dir + 'point_clouds/' + nm + ".particles", pc[0].detach().cpu().numpy())
		MSE = torch.mean((pred_y - corr.to(model.device))**2).item()
		CD = losses.CD(pred_y, pc.to(model.device)).detach().cpu().item()
		vib_CDs.append(CD)
		vib_MSES.append(MSE)
		y_log_var = y[1].reshape((num_models, batch_size,) + y[1].shape[1:])
		ale_unc = np.mean(np.exp(y_log_var.detach().cpu().numpy()), axis=0)
		ale_uncs.append(np.sum(ale_unc))
		np.savetxt(pred_dir + 'aleatoric/' + nm + ".npy", ale_unc.reshape((num_particles,3)))
		# Outlier degrees 
		img_out.append(data_info[nm]["image"])
		shape_out.append(data_info[nm]["shape"])
		# Record individual results
		individual_results[nm] = data_info[nm]
		individual_results[nm]["MSE"] = str(MSE) 
		individual_results[nm]["CD"] = str(CD)
		individual_results[nm]["aleatoric"] = str(np.sum(ale_unc))
		individual_results[nm]["epistemic"] = str(np.sum(epi_unc))
	results = {}
	print(name + " results: ")
	print("Mean " + name + "  det CD = " +str(round(np.mean(det_CDs),4))+'+-'+str(round(np.std(det_CDs), 4)))
	results["detCD"] = {"mean":str(round(np.mean(det_CDs),4)), "std":str(round(np.std(det_CDs), 4))}
	print("Mean " + name + "  VIB CD = " +str(round(np.mean(vib_CDs),4))+'+-'+str(round(np.std(vib_CDs), 4)))
	results["vibCD"] = {"mean":str(round(np.mean(vib_CDs),4)), "std":str(round(np.std(vib_CDs), 4))}
	print("Mean " + name + "  det MSE = " +str(round(np.mean(det_MSES),4))+'+-'+str(round(np.std(det_MSES), 4)))
	results["detMSE"] = {"mean":str(round(np.mean(det_MSES),4)), "std":str(round(np.std(det_MSES), 4))}
	print("Mean " + name + "  VIB MSE = " +str(round(np.mean(vib_MSES),4))+'+-'+str(round(np.std(vib_MSES), 4)))
	results["vibMSE"] = {"mean":str(round(np.mean(vib_MSES),4)), "std":str(round(np.std(vib_MSES), 4))}

	corr, _ = pearsonr(vib_CDs, ale_uncs)
	results["ale/CD"] = str(corr)
	print("Aleatoric uncertainty / CD correlation:", corr)
	corr, _ = pearsonr(vib_CDs, epi_uncs)
	results["epi/CD"] = str(corr)
	print("Epistemic uncertainty / CD correlation:", corr)
	corr, _ = pearsonr(vib_CDs, [sum(x) for x in zip(ale_uncs, epi_uncs)])
	results["pred/CD"] = str(corr)
	print("Predictive uncertainty / CD correlation:", corr)

	corr, _ = pearsonr(vib_MSES, ale_uncs)
	results["ale/MSE"] = str(corr)
	print("Aleatoric uncertainty / MSE correlation:", corr)
	corr, _ = pearsonr(vib_MSES, epi_uncs)
	results["epi/MSE"] = str(corr)
	print("Epistemic uncertainty / MSE correlation:", corr)
	corr, _ = pearsonr(vib_MSES, [sum(x) for x in zip(ale_uncs, epi_uncs)])
	results["pred/MSE"] = str(corr)
	print("Predictive uncertainty / MSE correlation:", corr)
	corr, _ = pearsonr(ale_uncs, img_out)
	results["ale/out"] = str(corr)
	print("Aleatoric uncertainty / Image outlier correlation:", corr)
	corr, _ = pearsonr(epi_uncs, shape_out)
	print("Epsitemic uncertainty / Shape outlier correlation:", corr)
	print("Aleatoric magnitude:", round(np.mean(ale_uncs),4), "+-", np.std(ale_uncs))
	results["aleatoric"] = {"mean":str(round(np.mean(ale_uncs),4)), "std":str(round(np.std(ale_uncs), 4))}
	print("Epistmeic magnitude:", np.mean(epi_uncs), "+-", np.std(epi_uncs))
	results["epistemic"] = {"mean":str(round(np.mean(epi_uncs),4)), "std":str(round(np.std(epi_uncs), 4))}
	with open(pred_dir+name+"_overall_results.json", "w") as outfile:
		json.dump(results, outfile, indent=2)
	with open(pred_dir+name+"_individual_results.json", "w") as outfile:
		json.dump(individual_results, outfile, indent=2)

def be_conc_test_loader(model, name, loader, pred_dir, samples, parameters):
	# print("Mixup:")
	# print(model.encoder.ConvolutionalBackbone.conv_blocks[0].conv_block[0].mixup)
	# print(model.encoder.ConvolutionalBackbone.fc_blocks[0].fc_block[1].mixup)
	# print(model.encoder.pred_z_dist.mixup)
	# print(model.decoder.fc_blocks[0].fc_block[0].mixup)
	# print(model.decoder.pred_y.mixup)
	num_models = parameters['batch_ensemble']['num_models']
	det_MSES = []
	vib_MSES = []
	det_CDs, vib_CDs = [],[]
	MSES= []
	ale_uncs = []
	epi_uncs = []
	img_out = []
	shape_out = []
	individual_results = {}
	with open(DATA_DIR + parameters['dataset'] + "/data_info.json") as json_file:
		data_info = json.load(json_file) 
	for img, pc, corr, nm in loader:
		# Deterministic z
		batch_size = img.shape[0]
		img = torch.cat([img for i in range(num_models)], dim=0)
		corr = corr.flatten(start_dim=1)
		_, y, _ =  model(img.to(model.device), 0, use_dropout=False)

		pred_y = torch.mean(y[0].reshape((num_models, batch_size,) + y[0].shape[1:]), axis=0)
		det_MSES.append(torch.mean(((pred_y - corr.to(model.device))**2)/batch_size).item())
		det_CDs.append(losses.CD(pred_y, pc.to(model.device)).detach().cpu())

		# # np.savetxt(pred_dir + 'particles/' + nm[0] + ".particles", pred_y.detach().cpu().numpy().reshape((128,3)))
		# epi_unc = torch.var(y[0], axis=0).detach().cpu().numpy()
		# epi_uncs.append(np.sum(epi_unc))
		# np.savetxt(pred_dir + 'epistemic/' + nm[0] + ".npy", epi_unc.reshape((128,3)))

		# Avg over z samples
		_, y, _ = model(img.to(model.device), samples, use_dropout=False)
		y[0] = y[0].reshape((num_models, batch_size,) + y[0].shape[1:])
		y[1] = y[1].reshape((num_models, batch_size,) + y[1].shape[1:])
		pred_y = torch.mean(y[0], axis=0)
		vib_MSES.append(torch.mean(((pred_y - corr.to(model.device))**2)/batch_size).item())
		vib_CDs.append(losses.CD(pred_y, pc.to(model.device)).detach().cpu())
		ale_unc = np.mean(np.exp(y[1].detach().cpu().numpy()), axis=0)

		# np.savetxt(pred_dir + 'aleatoric/' + nm[0] + ".npy", ale_unc.reshape((128,3)))
		ale_uncs.append(np.sum(ale_unc))

		# Avg over z_samples and dropout masks
		sampled_y_mus = []
		all_sampled_y_mus = []
		for i in range(samples):
			_, y, _ =  model(img.to(model.device), 0, use_dropout=True)
			all_sampled_y_mus.append(y[0].detach().cpu().numpy())
			pred_y = torch.mean(y[0].reshape((num_models, batch_size,) + y[0].shape[1:]), axis=0)
			sampled_y_mus.append(pred_y.detach().cpu().numpy())
		y_mu = np.mean(np.array(sampled_y_mus), axis=0)
		# np.savetxt(pred_dir + 'epistemic/' + nm[0] + ".npy", np.var(np.array(sampled_y_mus), axis=0).reshape((num_particles,3)))
		true_y = corr.detach().cpu().numpy()
		MSES.append(np.mean((y_mu - true_y)**2)/true_y.shape[0])
		# Correlation
		epi_uncs.append(np.sum(np.var(np.array(all_sampled_y_mus), axis=0)))

		# Correlation
		img_out.append(data_info[nm[0]]["image"])
		shape_out.append(data_info[nm[0]]["shape"])

	results = {}
	print()
	print("Mean " + name + "  VIB MSE = " +str(round(np.mean(vib_MSES),4))+'+-'+str(round(np.std(vib_MSES), 4)))
	results["vibMSE"] = {"mean":str(round(np.mean(vib_MSES),4)), "std":str(round(np.std(vib_MSES), 4))}
	print("Mean " + name + "  MSE = " +str(round(np.mean(MSES),4))+'+-'+str(round(np.std(MSES), 4)))
	results["MSE"] = {"mean":str(round(np.mean(MSES),4)), "std":str(round(np.std(MSES), 4))}
	corr, _ = pearsonr(vib_MSES, ale_uncs)
	results["ale/MSE"] = str(corr)
	print("Aleatoric uncertainty / MSE correlation:", corr)
	corr, _ = pearsonr(vib_MSES, epi_uncs)
	results["epi/MSE"] = str(corr)
	print("Epistemic uncertainty / MSE correlation:", corr)
	corr, _ = pearsonr(vib_MSES, [sum(x) for x in zip(ale_uncs, epi_uncs)])
	results["pred/MSE"] = str(corr)
	print("Predictive uncertainty / MSE correlation:", corr)
	corr, _ = pearsonr(ale_uncs, img_out)
	results["ale/out"] = str(corr)
	print("Aleatoric uncertainty / Image outlier correlation:", corr)
	corr, _ = pearsonr(epi_uncs, shape_out)
	results["epi/out"] = str(corr)
	print("Epsitemic uncertainty / Shape outlier correlation:", corr)
	print("Aleatoric magnitude:", round(np.mean(ale_uncs),4), "+-", np.std(ale_uncs))
	results["aleatoric"] = {"mean":str(round(np.mean(ale_uncs),4)), "std":str(round(np.std(ale_uncs), 4))}
	print("Epistmeic magnitude:", np.mean(epi_uncs), "+-", np.std(epi_uncs))
	results["epistemic"] = {"mean":str(round(np.mean(epi_uncs),4)), "std":str(round(np.std(epi_uncs), 4))}
	with open(pred_dir+name+"_results.json", "w") as outfile:
		json.dump(results, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-n', '--num_samples', help='number of samples ot use in testing', default=30)
    arg = parser.parse_args()
    for loader_to_test in ['train','combo_test']:
    	test(arg.config, loader_to_test, int(arg.num_samples))