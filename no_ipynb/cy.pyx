#cython: language_level=3
# --compile-args="-O2"

cimport cython
import numpy as np

cimport numpy as cnp
from numpy cimport npy_intp, npy_longdouble
from libcpp cimport bool
from libc.math cimport exp



@cython.boundscheck(False)
cdef int calc_e_jk(npy_intp site,
                   cnp.int32_t[::1] spins,          # the field
                   cnp.int32_t[:, :] neighbors):    # associations: neighbors[site, :] are local neighb sites
    """Calculate e_jk: the energy change for spins[site] -> *= -1"""
    
    cdef:
        int s_jk = spins[site]
        int summa = 0
        npy_intp prev = neighbors[site, 5]
        npy_intp curr
           
    
    for j in range(6):   
        curr = neighbors[site, j]
        summa += spins[curr] * spins[prev]
        prev = curr
    
    return -s_jk * summa


@cython.cdivision(True)
cdef bint mc_choice(int e_jk, double T, npy_longdouble uni):
    """принимаем или не принимаем переворот спина?"""
    cdef double r
    r = exp(2*e_jk/T)
    if e_jk > 0:
        return True
    elif uni <= r:
        return True
    else:
        return False


@cython.boundscheck(False) 
cdef void step(cnp.int32_t[::1] spins, cnp.int32_t[:, :] neigh, double T, npy_intp site, npy_longdouble uni):
    """крутим 1 спин"""

        
    cdef int L2, e_jk
    
    L2 = spins.shape[0]
    
    e_jk = calc_e_jk(site, spins, neigh)
    
    if mc_choice(e_jk, T, uni):
        spins[site] *= -1
        

@cython.boundscheck(False)
def mc_step(cnp.int32_t[::1] spins,
            cnp.int32_t[:, :] neighbors,
            double T):
    """perform L*L flips for 1 MC step"""
    
    cdef npy_intp num_steps = spins.shape[0]
    cdef int _
    
    cdef cnp.ndarray[double,
                ndim=1,
                negative_indices=False,
                mode='c'] unis = np.random.uniform(size=num_steps)
    cdef cnp.ndarray[npy_intp,
                ndim=1,
                negative_indices=False,
                mode='c'] sites = np.random.randint(num_steps, size=num_steps)
    
    for _ in range(num_steps):
        step(spins, neighbors, T, sites[_], unis[_])
        