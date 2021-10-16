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


def print_coefs ():
	"""Prints the coefficient of the polynomials for some parameters using two means of calculation"""
	c = .3
	N = 4

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




def main ():
	print_coefs()

if __name__ == "__main__":
	main()
