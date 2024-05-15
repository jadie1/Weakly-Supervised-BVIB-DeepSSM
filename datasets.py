# Jadie Adams
import os
import argparse
import nrrd
import json
import random
import numpy as np
import pyvista as pv
from sklearn.decomposition import PCA
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
from skimage.exposure import match_histograms
from skimage.util import random_noise


DATA_DIR = 'data/'

'''
Reads .nrrd files and returns numpy array
Whitens/normalizes images
'''
def get_images(image_files, loader_dir, train=False):
	# get all images
	all_images = []
	for image_file in image_files:
		img, header = nrrd.read(image_file)
		all_images.append(img)
	all_images = np.array(all_images)
	# get mean and std
	mean_path = loader_dir + 'mean_img.npy'
	std_path = loader_dir + 'std_img.npy'
	if train:
		mean_image = np.mean(all_images)
		std_image = np.std(all_images)

		np.save(mean_path, mean_image)
		np.save(std_path, std_image)
	else:
		mean_image = np.load(mean_path)
		std_image = np.load(std_path)
	# normalize
	norm_images = []
	for image in all_images:
		norm_images.append([(image-mean_image)/std_image])
	return np.array(norm_images)

def get_augmented_images(imgs, factor):
	aug_imgs = np.zeros((factor,)+imgs.shape)
	aug_imgs[0] = imgs
	for i in range(1,factor):
		for j in range(imgs.shape[0]):
			k = random.randrange(imgs.shape[0])
			aug_imgs[i][j] = match_histograms(imgs[j], imgs[k], channel_axis=0)
	return aug_imgs

'''
Reads .particle files and returns numpy array
'''
def get_correspondence_points(point_files):
	points = []
	for point_file in point_files:
		points.append(np.loadtxt(point_file))
	return np.array(points)

'''
Reads .vtk files and returns numpy array of vertices
'''
def get_point_clouds_from_meshes(mesh_files, num_points=5000):
	all_point_clouds = []
	for mesh_file in mesh_files:
		all_point_clouds.append(np.array(pv.read(mesh_file).points)[:5000,:])
	return np.array(all_point_clouds)

'''
Reads .txt files and returns numpy array of PCA scores
'''
def get_pca_vectors(z_files):
	all_zs = []
	if os.path.exists(z_files[0]):
		print("PCA files have dimension", np.loadtxt(z_files[0]).shape[0])
		for z_file in z_files:
			all_zs.append(np.loadtxt(z_file))
		all_zs = np.array(all_zs)
	else:
		print("No PCA files, using zero placeholder.")
		all_zs = np.zeros((len(z_files), 1))
	return all_zs

'''
Class for SSM datasets that works with Pytorch DataLoader
Includes images, point clouds, correspondence points, and names
'''
class SSMdataset():
	# aug_imgs,pcs,corrs,zs,names,noise_aug
	def __init__(self, imgs, pcs, corrs, zs,  names, noise_aug=False):
		self.imgs = torch.FloatTensor(imgs)
		self.pcs = torch.FloatTensor(pcs)
		self.corrs = torch.FloatTensor(corrs)
		self.zs = torch.FloatTensor(zs)
		self.names = names
		self.noise_aug = noise_aug
	def __getitem__(self, index):
		# Histogram aug - randomly select one
		if len(self.imgs.shape) == 6:
			choice = random.randrange(self.imgs.shape[0])
			# print('Histogram', choice)
			x = self.imgs[choice][index]
		else:
			x = self.imgs[index]
		# Noise aug - up to 1%
		if self.noise_aug:
			var = random.random()/100
			# print('Noise', var)
			x = torch.FloatTensor(random_noise(x, mode='gaussian', mean=0, var=var, clip=True))
		pc = self.pcs[index]
		y = self.corrs[index]
		z = self.zs[index]
		name = self.names[index]
		return x, pc, y, z, name
	def __len__(self):
		return len(self.names)

'''
Get train dataset
'''
def get_dataset(data_name, group_name, size=1, histogram_aug=False, noise_aug=False):
	data_dir = DATA_DIR+data_name+'/'
	# Get names from json
	json_file = data_dir + 'data_info.json'
	with open(json_file) as json_f: 
		data_info = json.load(json_f)
	all_names = list(data_info.keys())
	names = []
	if group_name == "combo_test":
		group_names = ["test","shape_outlier_test","image_outlier_test"]
	else:
		group_names = [group_name]
	for name in all_names:
		if data_info[name]['group'] in group_names:
			names.append(name)
	# Select subset
	names = names[:int(len(names)*size)]
	# Get files
	image_files, mesh_files, corr_files, z_files = [], [], [], []
	for name in names:
		image_files.append(data_dir + 'images/' + name + '.nrrd')
		mesh_files.append(data_dir + 'meshes/' + name + '.vtk')
		corr_files.append(data_dir+ 'particles/' + name + '.particles')
		z_files.append(data_dir+ 'pca_scores/' + name + '.pca')
	# Get numpy data
	imgs = get_images(image_files, data_dir, group_name=='train')
	pcs = get_point_clouds_from_meshes(mesh_files)
	corrs = get_correspondence_points(corr_files)
	zs = get_pca_vectors(z_files)
	if histogram_aug and group_name == "train":
		aug_imgs = get_augmented_images(imgs, factor=9)
		dataset = SSMdataset(aug_imgs,pcs,corrs,zs,names,noise_aug)
	elif group_name == "train":
		dataset = SSMdataset(imgs,pcs,corrs,zs,names,noise_aug)
	else:
		dataset = SSMdataset(imgs,pcs,corrs,zs,names)
	return dataset