
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


class HiddenLaw:
    def draw (self, n_sample):
        raise NotImplementedError("A subclass of HiddenLaw should implement the daw method")

class NormalLaw (HiddenLaw):
    def draw (self, n_sample):
        return np.random.normal(0,1,n_sample)
    def get_tab_coefs (self, I):
        if I > 7:
            raise NotImplementedError("I don't know the coefficients of this polynomial above I=7")
        else:
            to_return = [
                [1,0,0,0,0,0,0],
                [0,1,0,0,0,0,0],
                [-1,0,1,0,0,0,0],
                [0,-3,0,1,0,0,0],
                [3,0,-6,0,1,0,0],
                [0,15,0,-10,0,1,0],
                [-15,0,45,0,-15,0,1]
            ]
            return np.array(to_return)[:I,:I] / np.array([np.sqrt(np.math.factorial(i)) for i in range(I)]).reshape((-1,1))
        
class UniformLaw (HiddenLaw):
    def draw (self, n_sample):
        return np.random.uniform(-1,1,n_sample)
    def get_tab_coefs (self, I):
        if I > 7:
            raise NotImplementedError("I don't know the coefficients of this polynomial above I=7")
        else:
            to_return = [
                [1,0,0,0,0,0,0],
                [0,1,0,0,0,0,0],
                [-1,0,3,0,0,0,0],
                [0,-3,0,5,0,0,0],
                [3,0,-30,0,35,0,0],
                [0,15,0,-70,0,63,0],
                [-5,0,105,0,-315,0,231]
            ]
            to_return = np.array(to_return) / np.array([1, 1, 2, 2, 8, 8, 16]).reshape((-1,1))
            return to_return[:I,:I] / np.array([np.sqrt(1/(2*i+1)) for i in range(I)]).reshape((-1,1))
        
def get_error2 (law, N):
    I = 7 # nombre de polynômes à utiliser

    # Tirer un tas d'échantillons
    samples = law.draw(N)

    # Calculer la matrice des produits scalaires
    px = np.stack([np.power(samples,i) for i in range(I)])
    scalars = px @ px.T / N

    # Effectuer gram-schmidt
    u = np.zeros((I,I))
    e = np.zeros((I,I))
    v = np.eye(I)
    for i in range(I):
        u[i] = v[i]
        for j in range(i):
            num = e[j:j+1] @ scalars @ v[i:i+1].T 
            denom = 1 # u[j:j+1] @ scalars @ u[j:j+1].T    # <- small speedup due to using e instead of u for the calculations.
            proj = num/denom * e[j]
            u[i] = u[i] - proj
        e[i] = u[i] / np.sqrt(u[i:i+1] @ scalars @ u[i:i+1].T)

    theo_e = law.get_tab_coefs(I) 
    error = np.sum(np.square(theo_e-e), axis=1)
    return error




def plot_cum_error (config, law, name):
    all_eN = np.linspace(np.math.log(300), np.math.log(300000), 16)
    all_N = np.floor(np.exp(all_eN)).astype(np.int32)
    all_error = []
    M = 6 if config.debug else 64
    for i in range(M):
        cur_error = [np.sum(get_error2(law, n)[:4]) for n in all_N]
        all_error.append(np.array(cur_error))
    all_error = np.stack(all_error)
    all_error = np.sqrt(np.mean(all_error, axis=0))
    all_x = [all_N[0], all_N[-1]]
    fac = .8 * all_error[0] / np.power(all_N[0], -1/2)
    all_y = [np.power(x, -1/2) * fac for x in all_x]
    plt.loglog(all_x, all_y, "k")
    plt.loglog(all_N, all_error, "o")
    plt.legend(["slope -1/2", "effective error"])
    plt.title("Polynômes de " + name)
    plt.xlabel("N")
    
    show_or_plot("arbitrary_cahos/error_{}.png".format(name), config)

def plot_all_cum_error (config):
    plot_cum_error(config, NormalLaw (), "Hermite")
    plot_cum_error(config, UniformLaw (), "Legendre")

def plot_errors (config):
    all_eN = np.linspace(np.math.log(300), np.math.log(300000), 16)
    all_N = np.floor(np.exp(all_eN)).astype(np.int32)
    all_error = []
    M = 6 if config.debug else 64
    for i in range(M):
        cur_error = [get_error2(n)[1:] for n in all_N]
        all_error.append(np.array(cur_error))
    all_error = np.stack(all_error)
    all_error = np.sqrt(np.mean(all_error, axis=0))

    all_x = [all_N[0], all_N[-1]]
    fac = all_error[0,0] / np.power(all_N[0], -1/2)
    all_y = [np.power(x, -1/2) * fac for x in all_x]
    plt.loglog(all_x, all_y, "k")

    all_x = [all_N[0], all_N[-1]]
    fac = all_error[0,-1] / np.power(all_N[0], -1/2)
    all_y = [np.power(x, -1/2) * fac for x in all_x]
    plt.loglog(all_x, all_y, "k")
    
    plt.loglog(all_N, all_error, "o")
    plt.legend(["slope -1/2", "slope -1/2"] + ["i="+str(i) for i in range(1,8)], bbox_to_anchor=(0., 1.1, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
    plt.title("Error on MC Hermit coefficients estimation")
    plt.xlabel("N")
    
    show_or_plot("arbitrary_cahos/full_error.png", config)

def main ():
    
    # config = Config(debug=True, show_plots=True, save_plots=False)
    config = Config(debug=False, show_plots=True, save_plots=True)

    if (False):
        law = NormalLaw ()
        N = 100
        I = 7 # nombre de polynômes à utiliser


        # Tirer un tas d'échantillons
        samples = law.draw(N)

        # Calculer la matrice des produits scalaires
        px = np.stack([np.power(samples,i) for i in range(I)])
        scalars = px @ px.T / N
        print(scalars)
        exit()

    plot_all_cum_error(config)

    # plot_errors (config)


if __name__ == "__main__":
    main()