import numpy as np

from scipy.signal import lfilter, lfilter_zi, iirfilter

def downsample_rt(x, m, l,
                  interp_m=None,
                  interp_l=None,
                  block_size=256,
                  causal=True):
    # Create filters if not provided.
    if interp_m is None:
        interp_m = iirfilter(order, 1.0 / m, btype='lowpass', ftype='butter')
    if interp_l is None:
        interp_l = iirfilter(order, 1.0 / l, btype='lowpass', ftype='butter')

    # Calc initial filter states.
    zinterp_m = lfilter_zi(*interp_m)
    zinterp_l = lfilter_zi(*interp_l)
    zinterp_m = np.zeros_like(zinterp_m)
    zinterp_l = np.zeros_like(zinterp_l)

    # Set initial downsampling offset states.
    down_offset = 0
    redown_offset = 0

    # Calc delay state.
    if causal:
        blockout_minlen, blockout_maxlen = calc_block_range(block_size, m, l)
        delay = block_size - blockout_minlen
        overflow = blockout_maxlen - block_size
        storage_len = delay + block_size + overflow
        storage = np.zeros(storage_len, dtype=np.float64)
        read_idx = 0
        write_idx = delay

    # Perform RT downsampling. Keep a noncausal version around as a sanity check.
    if causal:
        wav_down = []
    wav_noncausal = []
    for i in xrange(0, len(x), block_size):
        block = x[i:i + block_size]

        # Upsample
        xup = np.zeros((m, block.shape[0]), dtype=block.dtype)
        xup[0] = block
        xup = xup.T.flatten()
        
        # Interpolate
        xinterp, zinterp_m = lfilter(interp_m[0], interp_m[1], xup, zi=zinterp_m)
        xinterp_len = xinterp.shape[0]
        assert xinterp_len == block.shape[0] * m
        
        # Create block to downsample
        xedge = xinterp[down_offset:]
        extra = xedge.shape[0] % downsample
        if extra > 0:
            xedge = np.concatenate([xedge, np.zeros(downsample - extra, dtype=xedge.dtype)])
        assert xedge.shape[0] % downsample == 0

        # Manage edge conditions for downsampling next block
        down_offset = downsample - extra
        if down_offset == downsample:
            down_offset = 0

        # Downsample
        xdown = np.reshape(xedge, (-1, downsample))[:, 0]
        
        # Reupsample
        xreup = np.zeros((downsample, xdown.shape[0]))
        xreup[0] = xdown
        xreup = xreup.T.flatten()
        
        # Reinterpolate
        if xreup.shape[0] > 0:
            xreinterp, zinterp_l = lfilter(interp_l[0], interp_l[1], xreup, zi=zinterp_l)
        else:
            # Calling lfilter with an empty array has consequences for some reason
            xreinterp = np.zeros_like(xreup)
        xreinterp_len = xreinterp.shape[0]
        
        # Create block to redownsample
        xreedge = xreinterp[redown_offset:]
        extra = xreedge.shape[0] % m
        if extra > 0:
            xreedge = np.concatenate([xreedge, np.zeros(m - extra, dtype=np.float64)])
        assert xreedge.shape[0] % m == 0

        # Manage edge conditions for redownsampling next block
        redown_offset = m - extra
        if redown_offset == m:
            redown_offset = 0
        
        # Redownsample
        xrecons = np.reshape(xreedge, (-1, m))[:, 0]
        
        blockout = xrecons
        wav_noncausal.append(blockout)

        if causal:
            if blockout.size == storage_len and delay + overflow != 0:
                print storage_len
                print 'yay'

            write_idx = write_to_ring_buffer(storage, blockout, write_idx)
            read_idx, blockdel = read_from_ring_buffer(storage, block.shape[0], read_idx)
            wav_down.append(blockdel)

    if causal:
        # Alert us if our theoretical ranges were ever wrong.
        blockout_lens = [x.shape[0] for x in wav_noncausal[:-1]]
        if len(blockout_lens) > 0:
	        assert min(blockout_lens) >= blockout_minlen
	        assert max(blockout_lens) <= blockout_maxlen
        return np.concatenate(wav_down)
    else:
        return np.concatenate(wav_noncausal)


if __name__ == '__main__':
    import sys

    from scipy.io.wavfile import read as wavread
    from scipy.io.wavfile import write as wavwrite
    from scipy.signal import iirfilter, lfilter_zi

    from util import rationalize_real, calc_block_range, read_from_ring_buffer, write_to_ring_buffer

    (wav_fp, fsn, out_fp) = sys.argv[1:]
    order = 8
    A = range(1, 50)
    B = range(1, 100)
    block_size = 256

    # Load wav file.
    fso, wav = wavread(wav_fp)

    # Average multi-channel.
    wav_f = wav.astype(np.float64)
    if wav_f.ndim == 2:
        wav_f = np.mean(wav_f, axis=1)
    assert wav_f.ndim == 1

    # Normalize.
    wav_f /= 32767.0

    # Parse rates.
    fso, fsn = float(fso), float(fsn)
    assert fsn <= fso

    # Discretize rate ratio.
    upsample, downsample = rationalize_real(fsn / fso, A=A, B=B)[0]
    print 'Closest ratio {}/{}, fsn of {}'.format(upsample, downsample, fso * (float(upsample) / downsample))

    # Perform downsampling.
    wav_down = downsample_rt(wav_f, upsample, downsample, block_size=block_size)

    # Write file out.
    wav_out = (wav_down * 32767.0).astype(np.int16)
    wavwrite(out_fp, fso, wav_out)