"""
source ./travail/x3a/.venv/bin/activate
cd travail/x3a/ea1/
python L2_error.py
"""

import numpy as np
import scipy
import scipy.stats
import scipy.special

import polynomial

import matplotlib.pyplot as plt



def print_coefs ():
	"""Prints the coefficient of the polynomials for some parameters using two means of calculation"""
	c = .3
	N = 6

	print("Hermit coefs (direct and recursive):")
	he_app = polynomial.HermiteApproximation ()
	print([he_app.gamma(n, c, use_rec=False) for n in range(N)])
	he_app = polynomial.HermiteApproximation ()
	print([he_app.gamma(n, c, use_rec=True) for n in range(N)])

	print("Laguerre coefs (direct and recursive):")
	la_app = polynomial.LaguerreApproximation (2)
	print([la_app.gamma(n, c, use_rec=False) for n in range(N)])
	la_app = polynomial.LaguerreApproximation (2)
	print([la_app.gamma(n, c, use_rec=True) for n in range(N)])

	print("Jacobi coefs (direct and recursive):")
	ja_app = polynomial.JacobiApproximation (2, 5)
	print([ja_app.gamma(n, c, use_rec=False) for n in range(N)])
	ja_app = polynomial.JacobiApproximation (2, 5)
	print([ja_app.gamma(n, c, use_rec=True) for n in range(N)])


def plot_l2 (poly_app):
	N = 20
	all_n = list(range(1, N))
	all_l2 = poly_app.calc_l2(N, 0.3)[1:]
	plt.semilogy(np.log10(all_n), all_l2)
	# plt.loglog(all_n, all_l2)
	plt.show()

def plot_all_l2 ():
	plot_l2(polynomial.HermiteApproximation ())
	plot_l2(polynomial.LaguerreApproximation (0))
	plot_l2(polynomial.JacobiApproximation (3, 4))
	plot_l2(polynomial.LegendreApproximation ())

def plot_q_comp (all_ja, legend="", N=-1, debug=False):
	N = N if N > 0 else (10 if debug else 40)
	all_q = np.linspace(0.05, 0.95, 20)

	n_sample = int(1e4) if debug else int(1e5)
	all_l2 = np.stack([ja.calc_all_l2(N, ja.c_quantile(all_q), size=n_sample) for ja in all_ja], axis=0)

	plt.plot(all_q, all_l2.T, "--o")
	plt.legend(legend)
	plt.xlabel("Quantile q")
	plt.ylabel("L2 Error")
	plt.title("N="+str(N))
	plt.show()


def plot_jacobi_comp (debug=False):
	N = 10 if debug else 40
	all_q = np.linspace(0.05, 0.95, 20)

	n_sample = int(1e4) if debug else int(1e5)

	all_ja = [
		polynomial.JacobiApproximation (0, 0),
		polynomial.JacobiApproximation (-.5, -.5),
		polynomial.JacobiApproximation (10, 12),
		polynomial.JacobiApproximation (-.7, .3),
	]

	all_l2 = np.stack([ja.calc_all_l2(N, ja.c_quantile(all_q), size=n_sample) for ja in all_ja], axis=0)

	plt.plot(all_q, all_l2.T, "--o")
	plt.legend([ja.full_description() for ja in all_ja])
	plt.xlabel("Quantile q")
	plt.ylabel("L2 Error")
	plt.title("N="+str(N))
	plt.show()

def plot_cops_comp (debug=False):
	N = 10 if debug else 50
	all_q = np.linspace(0.05, 0.95, 20)

	n_sample = int(1e4) if debug else int(1e5)

	all_ja = [
		polynomial.HermiteApproximation (),
		polynomial.LegendreApproximation(),
		polynomial.JacobiApproximation (-.5, -.5),
		polynomial.JacobiApproximation (10, 12),
		polynomial.LaguerreApproximation(0),
		polynomial.LaguerreApproximation(3),
	]

	all_l2 = np.stack([ja.calc_all_l2(N, ja.c_quantile(all_q), size=n_sample) for ja in all_ja], axis=0)

	plt.plot(all_q, all_l2.T, "--o")
	plt.legend([ja.full_description() for ja in all_ja])
	plt.xlabel("Quantile q")
	plt.ylabel("L2 Error")
	plt.title("N="+str(N))
	plt.show()

def plot_error_ratio (debug=False):
	N = 10 if debug else 40
	all_q = np.linspace(0.05, 0.95, 20)

	n_sample = int(1e4) if debug else int(1e5)
	
	all_ja = [
		polynomial.HermiteApproximation (),
		polynomial.LegendreApproximation(),
		polynomial.JacobiApproximation (-.5, -.5),
		polynomial.JacobiApproximation (10, 12),
		polynomial.LaguerreApproximation(0),
		polynomial.LaguerreApproximation(3),
	]

	all_ratio = np.stack([ja.calc_all_l2(N, ja.c_quantile(all_q), size=n_sample) * ja.bound_inv(N, all_q) for ja in all_ja], axis=0)

	plt.plot(all_q, all_ratio.T, "--o")
	plt.legend([ja.full_description() for ja in all_ja])
	plt.xlabel("Quantile q")
	plt.ylabel("Ratio of L2 error and error estimate")
	plt.title("N="+str(N))
	plt.show()

def plot_small_quantile (debug=False):
	N = 10 if debug else 50
	all_q = np.linspace(1e-3, 1e-4, 20)

	n_sample = int(1e4) if debug else int(1e5)

	all_ja = [
		polynomial.HermiteApproximation (),
		polynomial.JacobiApproximation (3, 0),
		polynomial.JacobiApproximation (0, 3),
		polynomial.LaguerreApproximation(3),
	]

	all_l2 = np.stack([ja.calc_all_l2(N, ja.c_quantile(all_q), size=n_sample) for ja in all_ja], axis=0)

	plt.plot(all_q, all_l2.T, "--o")
	plt.legend([ja.full_description() for ja in all_ja])
	plt.xlabel("Quantile q")
	plt.ylabel("L2 Error")
	plt.title("N="+str(N))
	plt.show()

def plot_high_quantile (debug=False):
	N = 10 if debug else 50
	all_q = np.linspace(1-1e-3, 1-1e-4, 20)

	n_sample = int(1e4) if debug else int(1e5)

	all_ja = [
		polynomial.HermiteApproximation (),
		polynomial.JacobiApproximation (3, 0),
		polynomial.JacobiApproximation (0, 3),
		polynomial.LaguerreApproximation(3),
	]

	all_l2 = np.stack([ja.calc_all_l2(N, ja.c_quantile(all_q), size=n_sample) for ja in all_ja], axis=0)

	plt.plot(all_q, all_l2.T, "--o")
	plt.legend([ja.full_description() for ja in all_ja])
	plt.xlabel("Quantile q")
	plt.ylabel("L2 Error")
	plt.title("N="+str(N))
	plt.show()

def plot_test (debug=False):
	N = 10 if debug else 50
	all_q = np.linspace(1e-4, 1-1e-4, 20)

	n_sample = int(1e4) if debug else int(1e5)

	all_ja = [
		polynomial.HermiteApproximation (),
		polynomial.JacobiApproximation (3, 0),
		polynomial.JacobiApproximation (0, 3),
		polynomial.LaguerreApproximation(3),
	]

	all_l2 = np.stack([ja.calc_all_l2(N, ja.c_quantile(all_q), size=n_sample) for ja in all_ja], axis=0)

	plt.plot(all_q, all_l2.T, "--o")
	plt.legend([ja.full_description() for ja in all_ja])
	plt.xlabel("Quantile q")
	plt.ylabel("L2 Error")
	plt.title("N="+str(N))
	plt.show()


def main ():
	# print_coefs()
	# plot_all_l2 ()

	debug = False
	
	# plot_jacobi_comp(debug=debug)
	# plot_cops_comp (debug=debug)

	# plot_error_ratio (debug=debug)

	plot_small_quantile (debug=debug)
	plot_high_quantile (debug=debug)

	# plot_test (debug=debug)

if __name__ == "__main__":
	main()
