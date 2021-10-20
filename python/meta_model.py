"""
source ./travail/x3a/.venv/bin/activate
cd travail/x3a/ea1/python
python meta_model.py
"""

import numpy as np
import scipy
import scipy.stats
import scipy.special
import sys
import time

import portfolio_utils
import polynomial

import matplotlib.pyplot as plt

from multiprocessing import Pool
from scipy.stats.kde import gaussian_kde
from pathlib import Path
from os import path

from dataclasses import dataclass


@dataclass
class Config:
	debug: bool
	show_plots: bool
	save_plots: bool

def result_name (config):
	res_name = "results_debug" if config.debug else "results"
	return res_name

def show_or_plot (path_suffix, config):
	if config.save_plots:
		res_name = result_name(config)
		full_path = path.join(res_name, path_suffix)
		Path(full_path).parent.mkdir(parents=True, exist_ok=True)
		plt.savefig(full_path, dpi=300)
	
	if config.show_plots:
		plt.show()
	else:
		plt.clf()



def plot_hermite_step_approximation (all_c, config):
	he = polynomial.HermiteApproximation()

	all_N = [5, 20, 50, 80]
	N_max = max(all_N)

	X = np.linspace(-3, 3, 300)

	fX, pX = he.calc_approx(N_max+1, all_c, X)

	for i, c in enumerate(all_c):
		plt.plot(fX[i], "k")
		for N in all_N:
			plt.plot(pX[i,N].T)

		plt.legend(["indicator"] + ["I="+str(N) for N in all_N])
		show_or_plot(path.join("meta_model_misc", "hermite_step_approximation_c"+str(c)), config)

def plot_hermite_step_approximation_error (all_c, config):
	N_max = 10 if config.debug else 80
	all_N = list(range(1, N_max))

	n_sample = int(1e4) if config.debug else int(1e5)
	X = np.random.normal(size=n_sample)

	he = polynomial.HermiteApproximation()
	fX, pX = he.calc_approx(N_max, all_c, X)
	fX = np.expand_dims(fX, axis=1)
	error = np.mean(np.square(pX-fX),axis=2)[:,1:]

	for i, c in enumerate(all_c):
		
		plt.plot(np.log(all_N), np.log(error[i]))
		plt.plot([0,np.max(np.log(all_N))], [-2, -2-np.max(np.log(all_N))/2], "--", color="orange")
		plt.legend(["mse", "slope=-1/2"])
		plt.xlabel("c="+str(c))

		show_or_plot(path.join("meta_model_misc", "hermite_step_approximation_error_c"+str(c)), config)



def make_save_folders (portfolio, config):
	res_name = result_name(config)
	Path(path.join(res_name, portfolio.name)).mkdir(parents=True, exist_ok=True)


def create_protfolio_A (config):
	K = int(5e3) if config.debug else int(5e5)
	p = [0.01 for k in range(1,K+1)]
	rho = [0.1 for k in range(1,K+1)]
	l = [1/np.sqrt(k) for k in range(1,K+1)]

	name = "portfolioA"
	portfolio = portfolio_utils.Portfolio(name, p, rho, l)
	make_save_folders(portfolio, config)

	return portfolio

def create_protfolio_B (config):
	K = int(5e3) if config.debug else int(5e5)
	p = [0.01*(1+np.sin(16*np.pi*k/K)) + 0.001 for k in range(1,K+1)]
	rho = [np.random.uniform(0, 1/np.sqrt(10)) + 0.001 for k in range(1,K+1)]
	l = [np.ceil(5*k/K)**2 for k in range(1,K+1)]

	name = "portfolioB"
	portfolio = portfolio_utils.Portfolio(name, p, rho, l)
	make_save_folders(portfolio, config)

	return portfolio


def calc_epsilons_sample (args):
	portfolio, i, n_sample = args
	he = polynomial.HermiteApproximation()
	to_return = []
	for sample in range(n_sample//10):
		thresholds = portfolio.all_a.reshape((-1,1)) * np.random.normal(size=(portfolio.K, 10)) + portfolio.all_b.reshape((-1,1))
		to_return.append(np.sum(np.expand_dims(portfolio.all_l,axis=1) * he.calc_gamma(i, thresholds), axis=0))
	return np.concatenate(to_return)

def plot_epsilon_distribution (portfolio, all_i, config):
	n_sample = int(1e3) if config.debug else int(1e4)

	with Pool(8) as p:
		for i in all_i:
			all_epsilon = np.concatenate(p.map(calc_epsilons_sample, [(portfolio.sub_portfolio(), i, n_sample//8)]*8))
			
			plt.hist(all_epsilon, bins=100, density=True)

			m = portfolio.m(i)
			v = portfolio.s(i,i)
			all_x = np.linspace(np.min(all_epsilon), np.max(all_epsilon), 100)
			all_normal = scipy.stats.norm.pdf((all_x-m)/np.sqrt(v)) / np.sqrt(v)
			plt.plot(all_x, all_normal)
			
			plt.title("i="+str(i))
			show_or_plot(path.join(portfolio.name, "epsilon_distribution_i"+str(i)+".png"), config)

def sub_simulate_cond_L (args):
	Z, portfolio, n_parallel = args
	to_return = []
	for sample in range(n_parallel//10):
		to_return.append(portfolio.simulate_cond_L(Z, n_parallel))
	return np.concatenate(to_return)

def plot_conditionnal_L (portfolio, all_Z, all_I, config):
	n_sample = int(1e3) if config.debug else int(1e4)

	max_I = np.max(all_I)
	
	if config.debug:
		print("Calculation of gaussian epsilon parameters")
	m = np.array([portfolio.m(i) for i in range(max_I+1)])
	v = portfolio.s_matrix(max_I+1)

	with Pool(8) as p:
		for Z in all_Z:
			
			if config.debug:
				print("simulation of Lg")
			HeZ = np.array([np.polynomial.hermite.Hermite(np.eye(i+1)[i])(Z/np.sqrt(2)) / np.power(2, i/2) for i in range(max_I+1)]).reshape((1, -1))
			epsilon = np.random.multivariate_normal(m, v, n_sample)
			Li = epsilon * HeZ
			LI = np.cumsum(Li, axis=1)
			LI = LI[:,all_I]

			all_x = np.linspace(np.min(LI), np.max(LI), 100)

			if config.debug:
				print("simulation of L")
			data = np.concatenate(p.map(sub_simulate_cond_L, [(Z, portfolio.sub_portfolio(), n_sample//8)]*8), axis=0)
			kde = gaussian_kde(data)
			plt.plot(all_x, kde(all_x))

			for data in LI.T:
				kde = gaussian_kde(data)
				plt.plot(all_x, kde(all_x))
			
			plt.legend(["L"] + ["I="+str(I) for I in all_I])
			plt.title("Z=" + str(Z))

			show_or_plot(path.join(portfolio.name, "conditionnal_L_Z"+str(Z)), config)

def sub_simulate_L (args):
	portfolio, n_parallel = args
	to_return = []
	for sample in range(n_parallel//10):
		to_return.append(portfolio.simulate_L(10))
	return np.concatenate(to_return)

def generate_L_distribution (portfolio, config):
	n_sample = 10000 if config.debug else int(1e5)
	start = time.time()
	with Pool(8) as p:
		data = np.concatenate(p.map(sub_simulate_L, [(portfolio.sub_portfolio(), n_sample//8)]*8), axis=0)
	if config.debug:
		print("Time to simulate L :", time.time()-start, "s")
	np.save(path.join(result_name(config), portfolio.name, "L.npy"), data)

def plot_L_distribution (portfolio, config):
	data = np.load(path.join(result_name(config), portfolio.name, "L.npy"))
	kde = gaussian_kde(data)

	all_x = np.linspace(np.min(data), np.max(data), 100)

	plt.hist(data, density=True, bins=50)
	plt.plot(all_x, kde(all_x))
	plt.legend(["L"])
	show_or_plot(path.join(portfolio.name, "L_histogram"), config)

def plot_gaussian_L_distribution (portfolio, config):
	n_sample = 10000 if config.debug else int(1e5)

	all_I = [1, 3, 6, 9]
	max_I = max(all_I)

	fac = np.array([np.power(2, -i/2) for i in range(max_I+1)])

	m = np.array([portfolio.m(i) for i in range(max_I+1)])
	v = portfolio.s_matrix(max_I+1)

	all_data = [[] for I in all_I]
	for sample in range(n_sample):
		alpha = np.random.multivariate_normal(m, v) * fac
		Z = np.random.normal()
		for data, I in zip(all_data, all_I):
			L = np.polynomial.hermite.Hermite(alpha[:I+1])(Z/np.sqrt(2))
			data.append(L)

	data = np.load(path.join(result_name(config), portfolio.name, "L.npy"))

	all_x = np.linspace(np.min(data), np.max(data), 100)

	kde = gaussian_kde(data)
	plt.plot(all_x, kde(all_x))

	for data, I in zip(all_data, all_I):
		kde = gaussian_kde(data)
		plt.plot(all_x, kde(all_x))

	plt.legend(["L"] + ["I="+str(I) for I in all_I])
	show_or_plot(path.join(portfolio.name, "L_LG"), config)

def qqplots (portfolio, config):

	L_data = np.load(path.join(result_name(config), portfolio.name, "L.npy"))
	L_data = np.sort(L_data)
	n_sample = L_data.shape[0]
	
	all_I = [1, 3, 6, 9]
	max_I = max(all_I)

	fac = np.array([np.power(2, -i/2) for i in range(max_I+1)])

	m = np.array([portfolio.m(i) for i in range(max_I+1)])
	v = portfolio.s_matrix(max_I+1)

	all_data = [[] for I in all_I]
	for sample in range(n_sample):
		alpha = np.random.multivariate_normal(m, v) * fac
		Z = np.random.normal()
		for data, I in zip(all_data, all_I):
			L = np.polynomial.hermite.Hermite(alpha[:I+1])(Z/np.sqrt(2))
			data.append(L)
	
	sorted_data = []
	for I, data in zip(all_I, all_data):
		L_g = np.sort(data)
		sorted_data.append(L_g)

		res = scipy.stats.linregress(L_g, L_data)
		x0 = np.min(L_data)
		x1 = np.max(L_data)
		y0 = x0*res.slope + res.intercept
		y1 = x1*res.slope + res.intercept

		plt.plot(L_data, sorted_data[-1], ".")
		plt.plot([x0,x1], [y0,y1], "k")
		plt.title("I="+str(I))
		show_or_plot(path.join(portfolio.name, "qqplots_I"+str(I)), config)

	# measurements = np.random.normal(loc = 20, scale = 5, size=100)   
	# scipy.stats.probplot(measurements, dist="norm", plot=plt)
	# plt.show()

def main ():
	config = Config(debug=True, save_plots=True, show_plots=False)

	# plot_hermite_step_approximation ([-1, 0, 1, 2], config)
	# plot_hermite_step_approximation_error ([-1, 0, 1, 2], config)

	# portfolio = create_protfolio_A (config)
	# portfolio = create_protfolio_B (config)

	# plot_epsilon_distribution (portfolio, [1, 3, 6, 9], config)
	# plot_conditionnal_L (portfolio, [-1, 0, 1, 2], [1, 3, 6, 9], config)

	# generate_L_distribution(portfolio, config)
	# plot_L_distribution (portfolio, config)
	# plot_gaussian_L_distribution (portfolio, config)
	# qqplots (portfolio, config)


if __name__ == "__main__":
	main()
