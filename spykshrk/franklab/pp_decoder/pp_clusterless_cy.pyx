
import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

cimport cython

from Cython.Compiler.Options import get_directive_defaults

#directive_defaults = get_directive_defaults()
#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def calc_posterior_cy(np.ndarray[np.float_t, ndim=2, mode='c'] likelihoods,
                      np.ndarray[np.float_t, ndim=2, mode='c'] transition_mat,
                      int pos_num_bins, double pos_delta):
    """
    
    Args:
        likelihoods (np.array): The evaluated likelihood function per time bin, from calc_likelihood(...).
        transition_mat (np.array): The point process state transition matrix.
        pos_num_bins (np.array): 

    Returns (pd.DataFrame): The decoded posteriors per time bin estimating the animal's location.

    """
    print('Using cython version of posterior calculation')
    cdef np.ndarray[np.float_t, ndim=1, mode='c'] last_posterior = np.ones(pos_num_bins)

    cdef np.ndarray[np.float_t, ndim=2, mode='c'] posteriors = np.zeros((likelihoods.shape[0], likelihoods.shape[1]))

    cdef np.ndarray[np.float_t, ndim=1, mode='c'] temp = np.zeros(likelihoods.shape[1])

    cdef double* posteriors_buffer

    posteriors_buffer = posterior_loop(&likelihoods[0,0], &transition_mat[0,0],
                                       likelihoods.shape[1], likelihoods.shape[0], pos_delta)

    posteriors.data = <char*>posteriors_buffer

    #for like_ii, like in enumerate(likelihoods):
    # for like_ii in range(len(likelihoods)):
    #     temp = likelihoods[like_ii, 0:450]
    #     temp = temp * np.matmul(transition_mat, last_posterior)
    #     posteriors[like_ii, :] = temp / (temp.sum() * pos_delta)
    #     last_posterior = temp

    # copy observ DataFrame and replace with likelihoods, preserving other columns

    return posteriors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef double* posterior_loop(double* likelihoods, double* trans_mat, int num_pos, int num_steps, double pos_delta):
    #cdef np.ndarray[np.float_t, ndim=1, mode='c'] temp = np.zeros(num_pos)

    cdef double* temp = <double *> malloc(num_pos * sizeof(double))
    cdef double* last_posterior = <double *> malloc(num_pos * sizeof(double))
    cdef double* posteriors = <double *> malloc(num_pos * num_steps * sizeof(double))

    cdef double trans_sum
    cdef int xx, yy, row_ii
    cdef double post_sum

    # Multiply transition matrix with last posterior
    for row_ii in range(num_steps):

        for xx in range(num_pos):
            trans_sum = 0
            for yy in range(num_pos):
                pass
                trans_sum=trans_mat[xx * num_pos + yy] * last_posterior[xx]

            temp[xx] = trans_sum

        # (Element-wise) Multiply likelihood with prior
        for xx in range(num_pos):
            temp[xx] = likelihoods[xx + row_ii * num_pos] * temp[xx]

        # Calculate normalization factor
        for xx in range(num_pos):
            post_sum += temp[xx]

        for xx in range(num_pos):
            temp[xx] = temp[xx] / (post_sum * pos_delta)

        posteriors[row_ii * num_pos: row_ii * num_pos + num_steps] = temp

    free(temp)
    free(last_posterior)


    return posteriors