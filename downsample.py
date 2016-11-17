import numpy as np

from scipy.signal import lfilter, lfilter_zi, iirfilter

from util import read_from_ring_buffer, write_to_ring_buffer, calc_block_range

def downsample_rt(x, m, l,
                  interp_m=None,
                  interp_l=None,
                  order=8,
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
            # TODO: remove this. Trying to verify my calculation for buffer size by seeing a case where we don't overshoot
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
    import argparse

    from scipy.io.wavfile import read as wavread
    from scipy.io.wavfile import write as wavwrite
    from scipy.signal import iirfilter, lfilter_zi

    from util import reduce_rational_list, rationalize_real

    parser = argparse.ArgumentParser()

    parser.add_argument('wav_fp', type=str, help='Input 16-bit signed PCM WAV filepath')
    parser.add_argument('fsn', type=float, help='Desired sampling rate')
    parser.add_argument('out_fp', type=str, help='Output 16-bit signed PCM WAV filepath')
    parser.add_argument('-a', '--numerators', type=str, help='CSV list of possible upsampling amounts')
    parser.add_argument('-b', '--denominators', type=str, help='CSV list of possible downsampling amounts')
    parser.add_argument('--iir_order', type=int, help='Order of IIR interpolation filters')
    parser.add_argument('--max_cascade', type=int, help='Maximum number of downsampling cascades')
    parser.add_argument('--block_size', type=int, help='Block size for downsampling, should only affect runtime and introduce delay in causal mode')
    parser.add_argument('--causal', dest='causal', action='store_true', help='Run the downsampling in real-time (using delay)')
    parser.add_argument('--noncausal', dest='causal', action='store_false', help='Run the downsampling offline (output length may not be equal to input length)')

    parser.set_defaults(
        a='0,1,2,3,5,7',
        b='1,2,3,5,7',
        iir_order=8,
        max_cascade=1,
        block_size=256,
        causal=False)

    args = parser.parse_args()
    A = [int(x) for x in args.a.split(',')]
    B = [int(x) for x in args.b.split(',')]

    # Load wav file.
    fso, wav = wavread(args.wav_fp)
    fso = float(fso)

    # Average multi-channel.
    wav_f = wav.astype(np.float64)
    if wav_f.ndim == 2:
        wav_f = np.mean(wav_f, axis=1)
    assert wav_f.ndim == 1

    # Normalize.
    wav_f /= 32767.0

    # Discretize rate ratio.
    rational_list = rationalize_real(args.fsn / fso, A=A, B=B, max_depth=args.max_cascade)
    print 'Closest rational list: {}, fsn of {}'.format(rational_list, fso * reduce_rational_list(rational_list))

    # Perform downsampling.
    wav_down = wav_f
    for upsample, downsample in rational_list:
        wav_down = downsample_rt(wav_down, upsample, downsample, block_size=args.block_size, causal=args.causal)

    # Write file out.
    wav_out = (wav_down * 32767.0).astype(np.int16)
    wavwrite(args.out_fp, fso, wav_out)