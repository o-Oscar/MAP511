"""
source ./travail/x3a/.venv/bin/activate
cd travail/x3a/ea1/python
python L2_error.py
"""

import numpy as np
import scipy
import scipy.stats
import scipy.special

import polynomial

import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path
from os import path

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
		plt.savefig(full_path, dpi=300, bbox_inches="tight")
	
	if config.show_plots:
		plt.show()
	else:
		plt.clf()


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


def plot_l2 (poly_app, all_c, all_slopes, config):
	N = 80
	all_n = list(range(1, N))

	n_sample = int(1e5)
	X = poly_app.random_sample(n_sample) # shape (sample)
	fX, pX = poly_app.calc_approx(N, all_c, X)
	fX = np.expand_dims(fX, axis=1)
	
	all_l2 = np.sqrt(np.mean(np.square(fX-pX), axis=2))[:,1:] # shape (all_c, degree)

	plt.plot(np.log(all_n), np.log(all_l2.T), "--.")
	# plt.loglog(all_n, all_l2)

	y_start = [-1.25, -1.5]
	m = np.max(np.log(all_n))
	for y, slope in zip(y_start, all_slopes):
		plt.plot([0, m], [y, y+slope*m])

	plt.legend(["c="+str(c) for c in all_c] + ["slope="+str(slope) for slope in all_slopes])
	plt.title("L2 Error - " + poly_app.full_description())
	show_or_plot(path.join("chaos_expansions", "L2_"+poly_app.full_description()), config)

def plot_all_l2 (config):
	plot_l2(polynomial.HermiteApproximation (), [0.1, 1, 2], [-1/4], config)
	plot_l2(polynomial.LegendreApproximation (), [0., 0.4, 0.9], [-1/2], config)
	plot_l2(polynomial.LaguerreApproximation (0), [1, 2, 3], [-1/4], config)
	plot_l2(polynomial.LaguerreApproximation (3), [1, 2, 3], [-1/4], config)
	plot_l2(polynomial.JacobiApproximation (3, 4), [0., 0.4, 0.6], [-1/4, -1/2], config)
	plot_l2(polynomial.JacobiApproximation (12, 11), [0., 0.4, 0.6], [-1/4, -1/2], config)


def plot_jacobi_comp (config):
	N = 10 if config.debug else 40
	all_q = np.linspace(0.05, 0.95, 20)

	n_sample = int(1e4) if config.debug else int(1e5)

	all_ja = [
		polynomial.JacobiApproximation (0, 0),
		polynomial.JacobiApproximation (-.5, -.5),
		polynomial.JacobiApproximation (10, 12),
		polynomial.JacobiApproximation (-.7, .3),
	]

	all_l2 = np.stack([ja.calc_all_l2(N, ja.c_quantile(all_q), size=n_sample) for ja in all_ja], axis=0)

	plt.plot(all_q, all_l2.T, "--o")
	plt.legend([ja.full_description() for ja in all_ja], bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
	plt.xlabel("Quantile q")
	plt.ylabel("L2 Error")
	plt.title("N="+str(N))
	show_or_plot(path.join("chaos_expansions", "jacobi_error"), config)

def plot_cops_comp (N, config):
	all_q = np.linspace(0.05, 0.95, 20)

	n_sample = int(1e4) if config.debug else int(1e5)

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
	plt.legend([ja.full_description() for ja in all_ja], bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
	plt.xlabel("Quantile q")
	plt.ylabel("L2 Error")
	plt.title("N="+str(N))
	show_or_plot(path.join("chaos_expansions", "cops_error_N"+str(N)), config)

def plot_all_cops_comp (config):
	all_N = [10] if config.debug else [20, 30, 40, 50]
	for N in all_N:
		plot_cops_comp(N, config)


def plot_error_ratio (N, config):
	all_q = np.linspace(0.05, 0.95, 20)

	n_sample = int(1e4) if config.debug else int(1e5)
	
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
	plt.legend([ja.full_description() for ja in all_ja], bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

	plt.xlabel("Quantile q")
	plt.ylabel("Ratio of L2 error and error estimate")
	plt.title("N="+str(N))
	show_or_plot(path.join("chaos_expansions", "error_ratio_N"+str(N)), config)

def plot_all_error_ratio (config):
	all_N = [10] if config.debug else [20, 30, 40, 50]
	for N in all_N:
		plot_error_ratio(N, config)

def plot_small_quantile (config):
	N = 10 if config.debug else 50
	all_q = np.linspace(1e-3, 1e-4, 20)

	n_sample = int(1e4) if config.debug else int(1e5)

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
	show_or_plot(path.join("chaos_expansions", "small_quantile"), config)

def plot_high_quantile (config):
	N = 10 if config.debug else 50
	all_q = np.linspace(1-1e-3, 1-1e-4, 20)

	n_sample = int(1e4) if config.debug else int(1e5)

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
	show_or_plot(path.join("chaos_expansions", "high_quantile"), config)

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
	config = Config(debug=False, save_plots=True, show_plots=True)

	# plot_all_l2 (config)

	
	plot_jacobi_comp(config)
	plot_all_cops_comp (config)

	# plot_all_error_ratio (config)

	# plot_small_quantile (config)
	# plot_high_quantile (config)

	# plot_test (debug=debug)

if __name__ == "__main__":
	main()
