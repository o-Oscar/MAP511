
import numpy as np
import scipy
import scipy.stats
import scipy.special


class PolynomialApproximation:
	def __init__ (self):
		pass
		self.all_gamma = {}
		self.name = "Undefinded"
		self.string_params = []

	def gamma (self, n, c, use_rec=False):
		"""Return \gamma_n(c), the nth coefficient for the approximation of 1_{X>c}"""
		key = (n,c)
		if key not in self.all_gamma:
			self.all_gamma[key] = self.calc_gamma(n, c) if not use_rec or n < 2 else self.calc_gamma_rec (n, c)
		return self.all_gamma[key]
	
	def calc_gamma (self, n, c):
		raise NotImplementedError("calc_gamma has to be implemented by any subclass of PolynomialApproximation")

	def calc_gamma_rec (self, n, c):
		raise NotImplementedError("calc_gamma_rec has to be implemented by any subclass of PolynomialApproximation")

	def poly_func (self, n, c):
		raise NotImplementedError("poly_func has to be implemented by any subclass of PolynomialApproximation")

	def random_sample (self, size):
		raise NotImplementedError("random_sample has to be implemented by any subclass of PolynomialApproximation")

	def c_quantile (self, q):
		raise NotImplementedError("c_quantile has to be implemented by any subclass of PolynomialApproximation")

	def bound_inv (self, N, q):
		raise NotImplementedError("bound_inv has to be implemented by any subclass of PolynomialApproximation")

	def calc_l2 (self, N, c, size=int(1e5)):
		X = self.random_sample(size)
		fX = np.greater(X, c).astype(np.float32)
		
		all_poly_func = [self.poly_func(n) for n in range(N)]
		coefs = np.array([self.gamma(n, c, use_rec=False) for n in range(N)])
		monoX = np.stack([poly(X) * coef for poly, coef in zip(all_poly_func, coefs)], axis=0)
		pX = np.cumsum(monoX, axis=0)

		return np.sqrt(np.mean(np.square(fX-pX), axis=1))
	
	def calc_all_l2 (self, N, all_c, size=int(1e5)):

		X = self.random_sample(size) # shape (sample)
		fX, pX = self.calc_approx(N, all_c, X)
		pX = pX[:,-1]
		
		return np.sqrt(np.mean(np.square(fX-pX), axis=1)) # shape (all_c)
	
	def calc_approx (self, N, all_c, X):

		fX = np.stack([np.greater(X, c).astype(np.float32) for c in all_c], axis=0) # shape (all_c, sample)
		
		all_mono = [self.poly_func(n) for n in range(N)] # shape (degree)
		coefs = np.array([[self.gamma(n, c, use_rec=False) for n in range(N)] for c in all_c]) # shape (all_c, degree)
		monoX = np.stack([poly(X) for poly in all_mono]) # shape (degree, sample)
		
		# TODO : give the option to keep all intermediary polynomials in order to get the evolution of accuracy with rising degree
		# pX = coefs @ monoX # shape (all_c, sample)
		pX = np.einsum('ij,jk->ijk', coefs, monoX) # shape (all_c, degree, sample)
		pX = np.cumsum(pX, axis=1) # shape (all_c, degree, sample)

		return fX, pX
	
	def params_description (self, to_join="   "):
		return to_join.join(self.string_params)

	def full_description (self):
		param_desc = self.params_description(to_join="_")
		return self.name + ("_" + param_desc if len(param_desc) > 0 else "")

class HermiteApproximation (PolynomialApproximation):
	def __init__ (self):
		super().__init__()
		self.norm = scipy.stats.norm

		self.name = "Hermite"
		self.string_params = []
	
	def hermite (self, n, c):
		he = np.polynomial.hermite.Hermite(np.eye(n+1)[n])
		return he(c/np.sqrt(2)) / np.power(2, n/2)

	def calc_gamma (self, n, c):
		if n == 0:
			return self.norm.cdf(-c)
		elif n == 1:
			return self.norm.pdf(c)
		else:
			return self.norm.pdf(c) * self.hermite(n-1, c) / np.math.factorial(n)

	def calc_gamma_rec (self, n2, c):
		n = n2-2
		return c / (n+2) * self.gamma(n+1, c) - n/(n+1)/(n+2) * self.gamma(n, c)

	def poly_func (self, n):
		poly = np.polynomial.hermite.Hermite(np.eye(n+1)[n])
		return lambda X: poly(X/np.sqrt(2)) / np.power(2, n/2)
	
	def random_sample (self, size):
		return np.random.normal(loc=0.0, scale=1.0, size=size)

	def c_quantile (self, q):
		return self.norm.ppf(q)

	def bound_inv (self, N, q):
		return np.exp(self.c_quantile(q)**2/4) * np.power(N, 1/4)

class LaguerreApproximation (PolynomialApproximation):
	def __init__ (self, alpha):
		super().__init__()
		self.alpha = alpha

		self.name = "Laguerre"
		self.string_params = ["a=" + str(alpha)]

	def calc_gamma (self, n, c):
		if n == 0:
			return scipy.special.gammaincc(self.alpha+1, c)
		elif n == 1:
			return -np.power(c, self.alpha+1) * np.exp(-c) / scipy.special.gamma(self.alpha+2)
		else:
			la = scipy.special.genlaguerre(n-1, self.alpha+1)
			return -np.math.factorial(n-1) * np.power(c, self.alpha+1) * np.exp(-c) * la(c) / scipy.special.gamma(n + self.alpha+1)

	def calc_gamma_rec (self, n2, c):
		n = n2-2
		return (2*n+self.alpha+2-c) / (n+self.alpha+2) * self.gamma(n+1, c) - n/(n+self.alpha+2) * self.gamma(n, c)

	def poly_func (self, n):
		return scipy.special.genlaguerre(n, self.alpha)
	
	def random_sample (self, size):
		return np.random.gamma(self.alpha+1, 1, size=size)

	def c_quantile (self, q):
		return scipy.stats.gamma.ppf(q, self.alpha+1)
		
	def bound_inv (self, N, q):
		return np.power(self.c_quantile(q), -self.alpha/2-1/4) * np.exp(self.c_quantile(q)/2) * np.power(N, 1/4)

class JacobiApproximation (PolynomialApproximation):
	def __init__ (self, alpha, beta):
		super().__init__()
		self.alpha = alpha
		self.beta = beta

		self.name = "Jacobi"
		self.string_params = ["a=" + str(alpha), "b=" + str(beta)]

	def inc_beta (self, a, b, x):
		return scipy.special.betainc(a, b, x) * scipy.special.beta(a, b)

	def calc_gamma (self, n, c):
		if n == 0:
			return scipy.special.betainc(self.alpha+1, self.beta+1, (1-c)/2)
		elif n == 1:
			fac = (self.alpha + self.beta + 3) / (self.alpha + 1) / (self.beta + 1) / self.inc_beta(self.alpha+1, self.beta+1, 1)
			return fac * ((self.alpha+1)*self.inc_beta(self.alpha+1, self.beta+1, (1-c)/2) - (self.alpha+self.beta+2)*self.inc_beta(self.alpha+2, self.beta+1, (1-c)/2))
		else:
			ja = scipy.special.jacobi(n-1, self.alpha+1, self.beta+1)
			fac1 = (2*n+self.alpha+self.beta+1) * np.math.factorial(n-1) * np.power(1-c, self.alpha+1) * np.power(1+c, self.beta+1) / 2**(self.alpha+self.beta+2)
			fac2 = scipy.special.gamma(n+self.alpha+self.beta+1) / scipy.special.gamma(n+self.alpha+1) / scipy.special.gamma(n+self.beta+1)
			return fac1 * fac2 * ja(c)

	def D (self, n, c):
		num = (self.alpha+self.beta+n+2) * (self.alpha+self.beta+2*n+5) * ((self.alpha-self.beta)*(self.alpha+self.beta+2)+c*(self.alpha+self.beta+2*n+2)*(self.alpha+self.beta+2*n+4))
		den = 2*(self.alpha+n+2)*(self.beta+n+2)*(self.alpha+self.beta+n+3)*(self.alpha+self.beta+2*n+2)
		return num/den
	
	def E (self, n, c):
		num = n*(self.alpha+self.beta+n+1)*(self.alpha+self.beta+n+2)*(self.alpha+self.beta+2*n+4)*(self.alpha+self.beta+2*n+5)
		den = (self.alpha+n+2)*(self.beta+n+2)*(self.alpha+self.beta+n+3)*(self.alpha+self.beta+2*n+1)*(self.alpha+self.beta+2*n+2)
		return -num/den
	
	def calc_gamma_rec (self, n2, c):
		n = n2-2
		return self.D(n, c) * self.gamma(n+1, c) + self.E(n, c) * self.gamma(n, c)

	def poly_func (self, n):
		return scipy.special.jacobi(n, self.alpha, self.beta)
	
	def random_sample (self, size):
		return 1 - 2*np.random.beta(self.alpha+1, self.beta+1, size=size)

	def c_quantile (self, q):
		return 2*scipy.stats.beta.ppf(q, self.alpha+1, self.beta+1) - 1

	def bound_inv (self, N, q):
		return np.power((1-self.c_quantile(q)), -self.alpha/2-1/4) * np.power((1+self.c_quantile(q)), -self.beta/2-1/4) * np.power(N, 1/2)

class LegendreApproximation (JacobiApproximation):
	def __init__ (self):
		super().__init__(0, 0)

		self.name = "Legendre"
		self.string_params = []

	def c_quantile (self, q):
		return 2*q-1

	def bound_inv (self, N, q):
		return np.power((1-self.c_quantile(q)**2), -1/4) * np.power(N, 1/2)