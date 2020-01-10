import copy
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys, os

sys.path.append('../')
from .characteristic_functions import charac_function, charac_function_multiscale
from .utils.graph_tools import laplacian


TAUS = [1, 10, 25, 50]
ORDER = 30
PROC = 'approximate'
ETA_MAX = 0.95
ETA_MIN = 0.80
NB_FILTERS = 2


def compute_cheb_coeff(scale, order):
    coeffs = [(-scale)**k * 1.0 / math.factorial(k) for k in range(order + 1)]
    return coeffs


def compute_cheb_coeff_basis(scale, order):
    xx = np.array([np.cos((2 * i - 1) * 1.0 / (2 * order) * math.pi)
                   for i in range(1, order + 1)])
    basis = [np.ones((1, order)), np.array(xx)]
    for k in range(order + 1-2):
        basis.append(2* np.multiply(xx, basis[-1]) - basis[-2])
    basis = np.vstack(basis)
    f = np.exp(-scale * (xx + 1))
    products = np.einsum("j,ij->ij", f, basis)
    coeffs = 2.0 / order * products.sum(1)
    coeffs[0] = coeffs[0] / 2
    return list(coeffs)


def heat_diffusion_ind(graph, taus=TAUS, order = ORDER, proc = PROC):
    '''
    This method computes the heat diffusion waves for each of the nodes
    INPUT:
    -----------------------
    graph    :    Graph (etworkx)
    taus     :    list of scales for the wavelets. The higher the tau,
                  the better the spread of the heat over the graph
    order    :    order of the polynomial approximation
    proc     :    which procedure to compute the signatures (approximate == that
                  is, with Chebychev approx -- or exact)

    OUTPUT:
    -----------------------
    heat     :     tensor of length  len(tau) x n_nodes x n_nodes
                   where heat[tau,:,u] is the wavelet for node u
                   at scale tau
    taus     :     the associated scales
    '''
    # Compute Laplacian
    a = nx.adjacency_matrix(graph)
    n_nodes, _ = a.shape
    thres = np.vectorize(lambda x : x if x > 1e-4 * 1.0 / n_nodes else 0)
    lap = laplacian(a)
    n_filters = len(taus)
    if proc == 'exact':
        ### Compute the exact signature
        lamb, U = np.linalg.eigh(lap.todense())
        heat = {}
        for i in range(n_filters):
             heat[i] = U.dot(np.diagflat(np.exp(- taus[i] * lamb).flatten())).dot(U.T)
    else:
        heat = {i: sc.sparse.csc_matrix((n_nodes, n_nodes)) for i in range(n_filters) }
        monome = {0: sc.sparse.eye(n_nodes), 1: lap - sc.sparse.eye(n_nodes)}
        for k in range(2, order + 1):
             monome[k] = 2 * (lap - sc.sparse.eye(n_nodes)).dot(monome[k-1]) - monome[k - 2]
        for i in range(n_filters):
            coeffs = compute_cheb_coeff_basis(taus[i], order)
            heat[i] = sc.sum([ coeffs[k] * monome[k] for k in range(0, order + 1)])
            temp = thres(heat[i].A) # cleans up the small coefficients
            heat[i] = sc.sparse.csc_matrix(temp)
    return heat, taus


def graphwave_alg(graph, time_pnts, taus= 'auto', 
              verbose=False, approximate_lambda=True,
              order=ORDER, proc=PROC, nb_filters=NB_FILTERS,
              **kwargs):
    ''' wrapper function for computing the structural signatures using GraphWave
    INPUT
    --------------------------------------------------------------------------------------
    graph             :   nx Graph
    time_pnts         :   time points at which to evaluate the characteristic function
    taus              :   list of scales that we are interested in. Alternatively,
                          'auto' for the automatic version of GraphWave
    verbose           :   the algorithm prints some of the hidden parameters
                          as it goes along
    approximate_lambda:   (boolean) should the range oflambda be approximated or
                          computed?
    proc              :   which procedure to compute the signatures (approximate == that
                          is, with Chebychev approx -- or exact)
    nb_filters        :   nuber of taus that we require if  taus=='auto'
    OUTPUT
    --------------------------------------------------------------------------------------
    chi               :  embedding of the function in Euclidean space
    heat_print        :  returns the actual embeddings of the nodes
    taus              :  returns the list of scales used.
    '''
    if taus == 'auto':
        if approximate_lambda is not True:
            a = nx.adjacency_matrix(graph)
            lap = laplacian(a)
            try:
                l1 = np.sort(sc.sparse.linalg.eigsh(lap, 2,  which='SM',return_eigenvectors=False))[1]
            except:
                l1 = np.sort(sc.sparse.linalg.eigsh(lap, 5,  which='SM',return_eigenvectors=False))[1]
        else:
            l1 = 1.0 / graph.number_of_nodes()
        smax = -np.log(ETA_MIN) * np.sqrt( 0.5 / l1)
        smin = -np.log(ETA_MAX) * np.sqrt( 0.5 / l1)
        taus = np.linspace(smin, smax, nb_filters)
    heat_print, _ = heat_diffusion_ind(graph, list(taus), order=order, proc = proc)
    chi = charac_function_multiscale(heat_print, time_pnts)
    return chi, heat_print, taus
