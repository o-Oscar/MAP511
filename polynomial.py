
import numpy as np
import scipy
import scipy.stats
import scipy.special


class PolynomialApproximation:
	def __init__ (self):
		pass
		self.all_gamma = {}

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

class HermiteApproximation (PolynomialApproximation):
	def __init__ (self):
		super().__init__()
		self.norm = scipy.stats.norm
	
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


class LaguerreApproximation (PolynomialApproximation):
	def __init__ (self, alpha):
		super().__init__()
		self.alpha = alpha

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

class JacobiApproximation (PolynomialApproximation):
	def __init__ (self, alpha, beta):
		super().__init__()
		self.alpha = alpha
		self.beta = beta

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
