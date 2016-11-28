import math

import numpy as np

def upsample_crude(x, m):
    assert m > 0

    xup = np.zeros((m, x.shape[0]), dtype=x.dtype)
    xup[0] = x
    xup = xup.T.flatten()

    assert xup.shape[0] == (x.shape[0] * m)

    return xup.T.flatten()

def downsample_crude(x, l):
    assert l > 0

    extra = x.shape[0] % l
    if extra > 0:
        x = np.concatenate([x, np.zeros(l - extra, dtype=x.dtype)])
    assert x.shape[0] % l == 0

    xdown = np.reshape(x, (-1, l))[:, 0]

    return extra, xdown


def write_to_ring_buffer(ring_buffer, samples, write_idx):
    """ Writes to a circular buffer (numpy arr).

    Parameters:
        ring_buffer: delay buffer (numpy arr)
        samples: samples to write
        write_idx: idx where to start writing
    Returns:
        write_idx: new write idx
    """
    assert samples.shape[0] <= ring_buffer.shape[0]
    for i in xrange(samples.shape[0]):
        if write_idx == ring_buffer.shape[0]:
            write_idx = 0
        ring_buffer[write_idx] = samples[i]
        write_idx += 1
    return write_idx

def read_from_ring_buffer(ring_buffer, nsamples, read_idx):
    """ Reads from a circular buffer (numpy arr).

    Parameters:
        ring_buffer: delay buffer (numpy arr)
        nsamples: number of samples to read
        read_idx: idx where to start reading
    Returns:
        read_idx: new read idx
        result: nsamples from the buffer
    """
    assert nsamples <= ring_buffer.shape[0]
    result = np.zeros(nsamples, dtype=ring_buffer.dtype)
    for i in xrange(nsamples):
        if read_idx == ring_buffer.shape[0]:
            read_idx = 0
        result[i] = ring_buffer[read_idx]
        read_idx += 1
    return read_idx, result

def reduce_rational_list(rat_list):
    map_rat = lambda (xn, xd): float(xn) / float(xd)
    reduce_rat = lambda x, y: x * y
    return reduce(reduce_rat, map(map_rat, rat_list))

def rationalize_real(rat, A, B, max_depth=1, curr_depth=0, curr_rational_list=[]):
    """Finds a recursive, rational approximation of ratio from integer lists A and B.

    This method is general but is intended to be used with A and B as short lists of primes.

    Parameters:
        rat: The ratio to approximate
        A: list of integer numerators
        B: list of integer denominators
        max_depth: Max recursive depth
        curr_depth: Current recursive depth
        curr_rational_list: Current list of rational approximations
    Returns:
        rational_list: List of tuples representing numerators and denominators
    """
    if curr_depth >= max_depth:
        return curr_rational_list

    closest = (float('inf'), (None, None))
    for a in A:
        for b in B:
            rational_list = rationalize_real(rat, A, B, max_depth=max_depth, curr_depth=curr_depth + 1, curr_rational_list=[(a, b)] + curr_rational_list[:])
            rat_approx = reduce_rational_list(rational_list)
            error = abs(rat_approx - rat)
            if error < closest[0]:
                closest = (error, rational_list)
    return closest[1]

def calc_block_range(n, ratios):
    ratios = ratios + [(b, a) for a, b in ratios[::-1]]

    b = [n, n]
    for l, m in ratios:
        b[0] = int(np.floor((l * float(b[0])) / m))
        b[1] = int(np.ceil((l * float(b[1])) / m))

    return tuple(b)