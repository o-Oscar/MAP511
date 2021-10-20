"""
source ./travail/x3a/.venv/bin/activate
cd travail/x3a/ea1/python
python portfolio.py
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




def compare_mu_calculators ():
	a = 1
	b = 1

	print("Mu coefficients with (1) recursive relation and (2) sum calculation")
	
	mu = portfolio_utils.MuCalculator (a, b)
	print([mu[i] for i in range(5)])
	mu = portfolio_utils.MuCalculatorSum (a, b)
	print([mu[i] for i in range(5)])

def compare_sigma_calculators ():
	a = 1
	b = 1

	# print("Sigma coefficients with (1) recursive relation and (2) sum calculation")
	print("Sigma coefficients")
	sigma = portfolio_utils.SigmaCalculator (a, b)
	np.set_printoptions(threshold=sys.maxsize)
	np.set_printoptions(precision=3)

	print(sigma.get_matrix(5))

def plot_step_app (all_c):
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
		plt.show()

def plot_step_err (debug=True):
	all_c = [-1, 0, 1, 2]
	N_max = 10 if debug else 80
	all_N = list(range(1, N_max))

	n_sample = int(1e4) if debug else int(1e5)
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
		# plt.ylabel("L2 Error")
		# plt.title("N="+str(N))
		plt.show()

def create_protfolio_A (K=int(1e5)):
	return portfolio_utils.SimplePortfolio(0.01, 0.1, [1/np.sqrt(k) for k in range(1,K+1)])

def calc_epsilons (args):
	i, K = args
	he = polynomial.HermiteApproximation()
	portfolio = portfolio_utils.SimplePortfolio(0.01, 0.1, [1/np.sqrt(k) for k in range(1,K+1)])
	return np.sum(np.expand_dims(portfolio.all_l,axis=1) * he.calc_gamma(i, portfolio.a*np.random.normal(size=(K,10))+portfolio.b), axis=0)

def plot_epsilon (all_i, debug=True):
	n_sample = int(1e3) if debug else int(1e4)
	K = int(5e5) if debug else int(5e5)

	for i in all_i:
		he = polynomial.HermiteApproximation()
		portfolio = create_protfolio_A(K)

		if False:
			all_epsilon = []
			for sample in range(n_sample//100):
				epsilon = np.sum(np.expand_dims(portfolio.all_l,axis=1) * he.calc_gamma(i, portfolio.a*np.random.normal(size=(K,100))+portfolio.b), axis=0)
				all_epsilon.append(epsilon)
			all_epsilon = np.concatenate(all_epsilon)
		else:
			with Pool(8) as p:
				all_epsilon = np.concatenate(p.map(calc_epsilons, [(i,K)]*(n_sample//10)))
			
		plt.hist(all_epsilon, bins=100, density=True)

		m = portfolio.m(i)
		v = portfolio.s(i,i)
		all_x = np.linspace(np.min(all_epsilon), np.max(all_epsilon), 100)
		all_normal = scipy.stats.norm.pdf((all_x-m)/np.sqrt(v)) / np.sqrt(v)
		plt.plot(all_x, all_normal)
		
		plt.title("i="+str(i))
		plt.show()


def sub_generate_epsilon_i (args):
	max_I, K = args
	portfolio = create_protfolio_A(K)
	epsilon_i = portfolio.simulate_epsilon_i(max_I+1)
	return epsilon_i

def generate_epsilon_i (max_I, debug=True):
	n_sample = int(1e4) if debug else int(1e4)
	K = int(5e3) if debug else int(5e5)
	with Pool(8) as p:
		all_epsilon = np.stack(p.map(sub_generate_epsilon_i, [(max_I,K)]*n_sample), axis=0)
	np.save("portfolioA/epsilon.npy", all_epsilon)


from scipy.stats.kde import gaussian_kde

def sub_simulate_cond_L (args):
	Z, K, n_parallel = args
	portfolio = create_protfolio_A(K)
	return portfolio.simulate_cond_L(Z, n_parallel)


def plot_LI (all_I, all_Z, debug=True):
	n_sample = int(1e2) if debug else int(1e4)
	K = int(5e5) if debug else int(5e5)
	max_I = max(all_I)
	all_epsilon = np.load("portfolioA/epsilon.npy")

	for Z in all_Z:
		
		HeZ = np.array([np.polynomial.hermite.Hermite(np.eye(i+1)[i])(Z/np.sqrt(2)) / np.power(2, i/2) for i in range(max_I+1)]).reshape((1, -1))
		Li = all_epsilon * HeZ
		LI = np.cumsum(Li, axis=1)
		LI = LI[:,all_I]

		with Pool(8) as p:
			data = np.concatenate(p.map(sub_simulate_cond_L, [(Z, K, 10)]*(n_sample//10)), axis=0)
		kde = gaussian_kde(data)

		# m = min(np.min(LI), np.min(data))
		# M = min(np.min(LI), np.min(data))
		# all_x = np.linspace(m, M, 50)
		all_x = np.linspace(np.min(LI), np.max(LI), 50)

		plt.plot(all_x, kde(all_x))

		for data in LI.T:
			kde = gaussian_kde(data)
			plt.plot(all_x, kde(all_x))
		
		plt.legend(["L"] + ["I="+str(I) for I in all_I])
		plt.title("Z=" + str(Z))
		plt.show()

def sub_simulate_L (args):
	K, n_parallel = args
	portfolio = create_protfolio_A(K)
	return portfolio.simulate_L(n_parallel)

def generate_distribution ():
	K = int(5e5)
	n_sample = 100000
	start = time.time()
	with Pool(8) as p:
		data = np.concatenate(p.map(sub_simulate_L, [(K, 10)]*(n_sample//10)), axis=0)
	print(time.time()-start)
	np.save("portfolioA/L.npy", data)

def plot_distribution ():
	data = np.load("portfolioA/L.npy")
	kde = gaussian_kde(data)

	all_x = np.linspace(np.min(data), np.max(data), 100)

	plt.hist(data, density=True, bins=50)
	plt.plot(all_x, kde(all_x))
	plt.legend(["L"])
	plt.show()

def plot_gaussian_distribution ():
	K = int(5e5)
	portfolio = create_protfolio_A(K)
	all_I = [1, 3, 6, 9]
	max_I = max(all_I)

	fac = np.array([np.power(2, -i/2) for i in range(max_I+1)])

	m = np.array([portfolio.m(i) for i in range(max_I+1)])
	v = portfolio.s_matrix(max_I+1)

	all_data = [[] for I in all_I]
	for sample in range(100000):
		alpha = np.random.multivariate_normal(m, v) * fac
		Z = np.random.normal()
		for data, I in zip(all_data, all_I):
			L = np.polynomial.hermite.Hermite(alpha[:I+1])(Z/np.sqrt(2))
			data.append(L)

	data = np.load("portfolioA/L.npy")

	all_x = np.linspace(np.min(data), np.max(data), 100)

	kde = gaussian_kde(data)
	plt.plot(all_x, kde(all_x))

	for data, I in zip(all_data, all_I):
		kde = gaussian_kde(data)
		plt.plot(all_x, kde(all_x))

	plt.legend(["L"] + ["I="+str(I) for I in all_I])
	plt.show()

def qqplots ():
	L_data = np.load("portfolioA/L.npy")
	L_data = np.sort(L_data)#[::1000]

	K = int(5e5)
	portfolio = create_protfolio_A(K)
	all_I = [1, 3, 6, 9]
	max_I = max(all_I)

	fac = np.array([np.power(2, -i/2) for i in range(max_I+1)])

	m = np.array([portfolio.m(i) for i in range(max_I+1)])
	v = portfolio.s_matrix(max_I+1)

	all_data = [[] for I in all_I]
	for sample in range(100000):
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
		plt.show()

	# measurements = np.random.normal(loc = 20, scale = 5, size=100)   
	# scipy.stats.probplot(measurements, dist="norm", plot=plt)
	# plt.show()



def main ():
	# compare_mu_calculators ()

	# compare_sigma_calculators ()

	# plot_step_app ([-1, 0, 1, 2])

	# plot_step_err (debug=False)

	# plot_epsilon([1, 3, 6, 9], False)

	# generate_epsilon_i(9, False)

	# plot_LI([1, 3, 6, 9], [-1, 0, 1, 2], False)

	# generate_distribution ()
	# plot_distribution ()
	# plot_gaussian_distribution ()

	qqplots ()

if __name__ == "__main__":
	main()
