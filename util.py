import math

import numpy as np

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

def calc_block_range(n, m, l):
    """Calculates the range [x, y] of possible block output sizes for a fractional downsample.

    Minimum delay is:    n - x
    Maximum overflow is: y - n
    Delay buffer len:    n - x + y

    n: block size
    m: upsample factor
    l: downsample factor
    """
    n = float(n)
    m = float(m)
    l = float(l)
    
    floor = lambda x: math.floor(x)
    ceil = lambda x: math.ceil(x)
    
    orig = [n, n]
    up = [m * orig[0], m * orig[1]]
    down = [floor(up[0] / l), ceil(up[1] / l)]
    reup = [l * down[0], l * down[1]]
    redown = [floor(reup[0] / m), ceil(reup[1] / m)]
    return int(redown[0]), int(redown[1])