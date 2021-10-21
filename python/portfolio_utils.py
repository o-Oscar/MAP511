
import numpy as np
from numpy import polynomial
from numpy.random.mtrand import normal
import scipy
import scipy.stats
import scipy.special

import polynomial
from multiprocessing import Pool

oo = np.inf

class MuCalculator:
	def __init__ (self, a, b):
		self.a = a
		self.b = b
		self.all_mu = {}
	
	def __getitem__ (self, n):
		if not type(n) == int:
			raise TypeError("Index should be of type n")
		if n < 0:
			raise TypeError("Index should positiv")
		
		if n not in self.all_mu:
			self.all_mu[n] = self.calc_mu(n)
		
		return self.all_mu[n]

	def calc_mu (self, i2):
		if i2 == 0:
			return scipy.stats.norm.cdf(-self.b/np.sqrt(1+self.a**2)) # TODO
		elif i2 == 1:
			return np.exp(-self.b**2/(2*(1+self.a**2))) / np.sqrt(2*np.pi) / np.sqrt(1+self.a**2)
		else:
			i = i2-2
			return self.b / i2 / (1+self.a**2) * self[i+1] - i / i2 / (i+1) /(1+self.a**2) * self[i]

class MuCalculatorSum (MuCalculator):
	def calc_mu (self, i):
		if i < 2:
			return super().calc_mu(i)
		else:
			fac = np.exp(-self.b**2/(2*(1+self.a**2))) / np.sqrt(2*np.pi) / i
			acc = 0
			for k in range((i-1)//2+1):
				n = i-1-2*k
				he = np.polynomial.hermite.Hermite(np.eye(n+1)[n]) / np.power(2, n/2)
				acc += self.a**(2*k) * he(self.b/np.sqrt(2)/(1+self.a**2)) / (2**k * np.math.factorial(k) * np.math.factorial(i-1-2*k) * np.power(1+self.a**2, k+1/2))
			return fac * acc


class SigmaCalculator:
	def __init__ (self, a, b, mu=None, mu_p=None):
		self.a = a
		self.b = b
		self.all_sigma = {}

		self.mu = MuCalculator(a, b) if mu is None else mu
		self.mu_p = MuCalculator(a/np.sqrt(1+a**2), b/(1+a**2)) if mu_p is None else mu_p

	def get_matrix (self, N):
		to_return = [[self[i,j] for j in range(N)] for i in range(N)]
		return np.array(to_return)

	def __getitem__ (self, key):
		if not type(key) == tuple:
			raise TypeError("Key should be of type tuple")
		if not len(key) == 2:
			raise TypeError("Key should be of length 2")
		
		i, j = key
		if not (type(i) == int and type(j) == int):
			raise TypeError("Index should be of type n")
		if i < 0 or j < 0:
			raise TypeError("Index should positiv")
		
		if key not in self.all_sigma:
			self.all_sigma[key] = self.calc_mu(i, j)
		
		return self.all_sigma[key]

	def calc_mu (self, i_init, j_init):
		if i_init == 0 and j_init == 0:
			Sigma = np.asarray([[1+self.a**2, self.a**2],[self.a**2, 1+self.a**2]])
			# denom = 2 * np.pi * np.sqrt(np.linalg.det(Sigma))

			# def func (x, y):
			# 	t = np.array([x, y]).reshape((-1, 1))
			# 	return np.exp(-.5 * t.T @ np.linalg.inv(Sigma) @ t)

			# first_res = scipy.integrate.dblquad(func, -oo, -self.b, lambda x: -oo, lambda x: -self.b)[0] / denom - self.mu[0]**2

			secound_res = scipy.stats.multivariate_normal([0,0], Sigma).cdf([-self.b,-self.b]) - self.mu[0]**2

			# print(first_res, secound_res)
			
			return secound_res
		
		elif i_init == 0 and j_init == 1:
			return self.mu[1] * (self.mu_p[0] - self.mu[0])
		
		elif i_init == 0:
			i = j_init-2
			acc = 0
			acc += self.b / (i+2) / (1+self.a**2) * self[0,i+1]
			acc -= i / (i+1) / (i+2) / (1+self.a**2) * self[0,i]
			acc -= self.a**2 / (i+2) / (1+self.a**2) * self.mu[1] * self.mu_p[i+1]
			return acc
		
		elif j_init == 0:
			return self[j_init,i_init]

		else:
			i = i_init-1
			j = j_init-1
			fac = 1 / self.a**2 / (i+1)
			acc = 0
			acc -= (1+self.a**2) * (j+2) * self[i,j+2]
			acc += self.b * self[i,j+1]
			acc -= j/(j+1) * self[i,j]
			return fac * acc - self.mu[i+1] * self.mu[j+1]


# TODO : implement the sigma calculation using sums

class Portfolio:
	def __init__ (self, name, all_p, all_rho, all_l, do_init_mu_sigma=True):
		self.name = name

		self.all_p = np.array(all_p)
		self.all_rho = np.array(all_rho)
		self.all_a = -np.sqrt(1-np.square(all_rho)) / np.abs(all_rho)
		self.all_b = -scipy.stats.norm.ppf(all_p) / np.abs(all_rho)
		
		self.K = len(all_l) if type(all_l) == list else all_l.shape[0]

		self.all_l = np.array(all_l)

		self.all_m = {}
		self.all_s = {}
		self.all_matrix = {}

		if do_init_mu_sigma:
			self.init_mu_sigma ()

	def init_mu_sigma (self):
		self.all_mu = [MuCalculator(a, b) for a, b in zip(self.all_a, self.all_b)]
		self.all_sigma = [SigmaCalculator(a, b) for a, b in zip(self.all_a, self.all_b)]

	def sub_portfolio (self):
		return Portfolio(self.name, self.all_p, self.all_rho, self.all_l, False)

	def m (self, i):
		if i not in self.all_m:
			self.all_m[i] = self.calc_m(i)
		return self.all_m[i]

	def s (self, i, j):
		key = (i,j)
		if key not in self.all_m:
			self.all_s[key] = self.calc_s(i, j)
		return self.all_s[key]
	
	def s_matrix (self, N):
		if not N in self.all_matrix:
			print("calculating sigma matrix")
			to_return = np.zeros((N,N))
			for l, a, b in zip(self.all_l, self.all_a, self.all_b):
				sigma = SigmaCalculator(a, b)
				to_return += l**2 * sigma.get_matrix(N)
			print("done")
			self.all_matrix[N] = to_return
		return self.all_matrix[N]
		# to_return = [[self.s(i,j) for j in range(N)] for i in range(N)]
		# return np.array(to_return)

	def calc_m (self, i):
		return np.sum([l * np.sign(rho) * mu[i] for l, rho, mu in zip(self.all_l, self.all_rho, self.all_mu)])
	
	def calc_s (self, i, j):
		return np.sum([l**2 * sigma[i,j] for l, sigma in zip(self.all_l, self.all_sigma)])
	
	def simulate_epsilon_i (self, I):
		cutoffs = self.a*np.random.normal(size=(self.K,))+self.b
		he = polynomial.HermiteApproximation()
		epsilon_i = np.stack([self.all_l * he.calc_gamma(i, cutoffs) * np.sign(self.rho)**i for i in range(I)]) # (I, K)
		return np.sum(epsilon_i, axis=1)

	def simulate_cond_L (self, Z, n_parallel):
		prob = 1-scipy.stats.norm.cdf((Z*np.sign(self.all_rho)-self.all_b)/self.all_a)
		Y = np.random.binomial(1, prob, size=(n_parallel, self.K))
		return np.sum(np.expand_dims(self.all_l, axis=0) * Y, axis=1)

	def simulate_L (self, n_parallel):
		Z = np.random.normal(size=(1, n_parallel))
		epsilon = np.random.normal(size=(self.K, n_parallel))
		default = (np.sign(self.all_rho).reshape((self.K, 1)) * Z) > (np.asarray(self.all_a).reshape((self.K, 1)) * epsilon + np.asarray(self.all_b).reshape((self.K, 1)))
		Y = default.astype(np.float32)

		L = np.sum(np.asarray(self.all_l).reshape((self.K, 1)) * Y, axis=0)
		return L

"""
class SimplePortfolio (Portfolio):
	def __init__ (self, p, rho, all_l):
		self.K = len(all_l)

		self.p = p
		self.all_p = [p] * self.K
		self.rho = rho
		self.all_rho = [rho] * self.K

		self.a = -np.sqrt(1-np.square(rho)) / np.abs(rho)
		self.b = -scipy.stats.norm.ppf(p) / np.abs(rho)
		self.all_a = [self.a] * self.K
		self.all_b = [self.b] * self.K


		self.all_l = all_l
		
		self.all_m = {}
		self.all_s = {}

		self.init_mu_sigma ()

	def init_mu_sigma (self):
		self.mu = MuCalculator(self.a, self.b)
		self.sigma = SigmaCalculator(self.a, self.b)

	def calc_m (self, i):
		return np.sum(self.all_l) * np.sign(self.rho)**i * self.mu[i]
	
	def calc_s (self, i, j):
		return np.sum(np.square(self.all_l)) * self.sigma[i,j]

	def simulate_LI (self, I, Z, n_parallel):
		HeZ = np.array([np.polynomial.hermite.Hermite(np.eye(i+1)[i])(Z/np.sqrt(2)) / np.power(2, i/2) for i in range(I)]).reshape((-1, 1))

		cutoffs = self.a*np.random.normal(size=(self.K, n_parallel))+self.b
		he = polynomial.HermiteApproximation()
		all_gamma = np.stack([he.calc_gamma(i, cutoffs) * np.sign(self.rho)**i for i in range(I)]) # (I, K, 10)
		epsilon_i = np.sum(np.expand_dims(self.all_l, axis=(0,2)) * all_gamma, axis=1) # (I, 10)
		return np.cumsum(epsilon_i * HeZ, axis=0)
"""