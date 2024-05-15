import os
import glob
import numpy as np
import json
from abc import ABC, abstractmethod

# abstract base class for embedders 
class Embedder(ABC):
	# abstract method
	def __init__(self, data_matrix):
		self.data_matrix = data_matrix
	def getEmbeddedMatrix(self):
		pass
	def project(self, PCA_instance):
		pass
 
class PCA_Embbeder():
	# overriding abstract methods
	def __init__(self, data_matrix, num_dim=0, percent_variability=0.95):
		self.data_matrix = data_matrix
		self.num_dim=num_dim
		self.percent_variability = percent_variability
	# run PCA on data_matrix for PCA_Embedder
	def run_PCA(self):
		# get covariance matrix (uses compact trick)
		N = self.data_matrix.shape[0]
		data_matrix_2d = self.data_matrix.reshape(self.data_matrix.shape[0], -1).T # flatten data instances and transpose
		mean = np.mean(data_matrix_2d, axis=1)
		centered_data_matrix_2d = (data_matrix_2d.T - mean).T
		trick_cov_matrix  = np.dot(centered_data_matrix_2d.T,centered_data_matrix_2d) * 1.0/np.sqrt(N-1)
		# get eignevectors and eigenvalues
		eigen_values, eigen_vectors = np.linalg.eigh(trick_cov_matrix)
		eigen_vectors = np.dot(centered_data_matrix_2d, eigen_vectors)
		for i in range(N):
			eigen_vectors[:,i] = eigen_vectors[:,i]/np.linalg.norm(eigen_vectors[:,i])
		eigen_values = np.flip(eigen_values)
		eigen_vectors = np.flip(eigen_vectors, 1)
		cumDst = np.cumsum(eigen_values) / np.sum(eigen_values)
		if self.num_dim == 0:
			num_dim = np.where(cumDst > float(self.percent_variability))[0][0] + 1
		else:
			num_dim = self.num_dim
		W = eigen_vectors[:, :num_dim]
		PCA_scores = np.matmul(centered_data_matrix_2d.T, W)
		print("The PCA modes of particles being retained : " + str(num_dim))
		print("Variablity preserved: " + str(float(cumDst[num_dim-1])))
		self.mean=mean
		self.num_dim = num_dim
		self.PCA_scores = PCA_scores
		self.cumDst = np.cumsum(eigen_values) / np.sum(eigen_values)
		self.eigen_vectors = eigen_vectors
		self.eigen_values = eigen_values
		return PCA_scores
	# write PCA info to files 
	def write_PCA(self, out_dir, suffix):
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		np.save(out_dir +  'original_PCA_scores.npy', self.PCA_scores)
		mean = np.mean(self.data_matrix, axis=0)
		np.savetxt(out_dir + 'mean.' + suffix, mean)
		np.savetxt(out_dir + 'eigenvalues.txt', self.eigen_values)
		for i in range(self.data_matrix.shape[0]):
			nm = out_dir + 'pcamode' + str(i) + '.' + suffix
			data = self.eigen_vectors[:, i]
			data = data.reshape(self.data_matrix.shape[1:])
			np.savetxt(nm, data)
		np.savetxt(out_dir + 'cummulative_variance.txt', self.cumDst)
	# returns embedded form of dtat_matrix
	def getEmbeddedMatrix(self, data_matrix):
		N = data_matrix.shape[0]
		data_matrix_2d = data_matrix.reshape(data_matrix.shape[0], -1).T # flatten data instances and transpose
		centered_data_matrix_2d = (data_matrix_2d.T - self.mean).T
		W = self.eigen_vectors[:, :self.num_dim]
		PCA_scores = np.matmul(centered_data_matrix_2d.T, W)
		return PCA_scores
	# projects embbed array into data
	def project(self, PCA_instance):
		W = self.eigen_vectors[:, :self.num_dim].T
		mean = np.mean(self.data_matrix, axis=0)
		data_instance =  np.matmul(PCA_instance, W) + mean.reshape(-1)
		data_instance = data_instance.reshape((-1,self.data_matrix.shape[1], self.data_matrix.shape[2]))
		return data_instance

'''
Reads data from files in given list and turns into one np matrix
'''
def create_data_matrix(file_list):
	data_matrix = []
	for file in file_list:
		data_matrix.append(np.loadtxt(file))
	return np.array(data_matrix)

def write_scores(out_dir, names, scores):
	os.makedirs(out_dir, exist_ok=True)
	for i in range(len(names)):
		np.savetxt(out_dir+names[i]+'.pca', scores[i])

if __name__ == '__main__':
	DATA_DIR = 'data/'
	dataset = 'liver'
	num_PCA = 115

	json_file = DATA_DIR + dataset+ '/data_info.json'
	with open(json_file) as json_f: 
		data_info = json.load(json_f)
	all_names = sorted(list(data_info.keys()))

	# PCA on training set 
	train_particle_files = []
	train_names = []
	print('train')
	for name in all_names:
		if data_info[name]['group'] == 'train':
			train_particle_files.append(DATA_DIR+dataset+'/particles/'+name+'.particles')
			train_names.append(name)
	train_point_matrix = create_data_matrix(train_particle_files)
	PointEmbedder = PCA_Embbeder(train_point_matrix, num_PCA)
	train_pca_scores = PointEmbedder.run_PCA()
	PointEmbedder.write_PCA(DATA_DIR+dataset+'/' + "PCA_Particle_Info/", "particles") # write PCA info for DeepSSM testing
	write_scores(DATA_DIR+dataset+'/pca_scores/', train_names, train_pca_scores)

	np.save(DATA_DIR + dataset+ '/mean_PCA.npy', np.mean(train_pca_scores))
	np.save(DATA_DIR + dataset+ '/std_PCA.npy', np.std(train_pca_scores))

	# PCA on held out sets
	groups = []
	for name in all_names:
		groups.append(data_info[name]['group'])
	groups = set(groups)
	groups.remove('train')
	for group in groups:
		print(group)
		particle_files, names = [], []
		for name in all_names:
			if data_info[name]['group'] == group:
				particle_files.append(DATA_DIR+dataset+'/particles/'+name+'.particles')
				names.append(name)
		point_matrix = create_data_matrix(particle_files)
		pca_scores = PointEmbedder.getEmbeddedMatrix(point_matrix)
		write_scores(DATA_DIR+dataset+'/pca_scores/', names, pca_scores)
		