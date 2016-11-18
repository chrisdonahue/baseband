import numpy as np

from scipy.signal import lfilter

from util import read_from_ring_buffer, write_to_ring_buffer, calc_block_range, upsample_crude, downsample_crude

def downsample_rt(x, m, l,
                  interp_m,
                  interp_l,
                  block_size=256,
                  causal=True):
    assert m >= 0 and l > 0
    if m == 0:
        return np.zeros((0,), dtype=x.dtype)

    # Calc initial filter states.
    zinterp_m = np.zeros(max(interp_m[0].shape[0], interp_m[1].shape[0]) - 1, dtype=np.float64)
    zinterp_l = np.zeros(max(interp_l[0].shape[0], interp_l[1].shape[0]) - 1, dtype=np.float64)

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
        if m > 0:
            xup = upsample_crude(block, m)
        else:
            xup = np.zeros((0,), dtype=block.dtype)
        
        # Interpolate (allow aliasing)
        if xup.shape[0] > 0:
            xinterp, zinterp_m = lfilter(interp_m[0], interp_m[1], xup, zi=zinterp_m)
        else:
            xinterp = np.zeros_like(xup)

        # Apply gain
        xinterp *= float(m)
        
        # Downsample
        extra, xdown = downsample_crude(xinterp[down_offset:], l)
        down_offset = l - extra
        if down_offset == l:
            down_offset = 0
        
        # Reupsample
        xreup = upsample_crude(xdown, l)
        
        # Reinterpolate (anti-alias)
        # TODO: NEED TO ANTI-ALIAS HERE
        if xreup.shape[0] > 0:
            xreinterp, zinterp_l = lfilter(interp_l[0], interp_l[1], xreup, zi=zinterp_l)
        else:
            # Calling lfilter with an empty array has consequences for some reason
            xreinterp = np.zeros_like(xreup)

        # Apply gain
        xreinterp *= float(l)

        # Redownsample
        reextra, xrecons = downsample_crude(xreinterp[redown_offset:], m)
        redown_offset = m - reextra
        if redown_offset == m:
            redown_offset = 0
        
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
    from scipy.signal import iirfilter, remez

    from util import reduce_rational_list, rationalize_real

    parser = argparse.ArgumentParser()

    parser.add_argument('wav_fp', type=str, help='Input 16-bit signed PCM WAV filepath')
    parser.add_argument('fsn', type=float, help='Desired sampling rate')
    parser.add_argument('out_fp', type=str, help='Output 16-bit signed PCM WAV filepath')
    parser.add_argument('--a', type=str, help='CSV list of possible upsampling amounts')
    parser.add_argument('--b', type=str, help='CSV list of possible downsampling amounts')
    parser.add_argument('--use_fir', dest='fir', action='store_true', help='Use FIR filters for interpolation')
    parser.add_argument('--use_iir', dest='fir', action='store_false', help='Use IIR filters for interpolation')
    parser.add_argument('--fir_ntaps', type=int, help='Ntaps for FIR')
    parser.add_argument('--fir_tol', type=float, help='Cutoff tolerance for FIR')
    parser.add_argument('--iir_order', type=int, help='Order of IIR interpolation filters')
    parser.add_argument('--max_cascade', type=int, help='Maximum number of downsampling cascades')
    parser.add_argument('--block_size', type=int, help='Block size for downsampling, should only affect runtime and introduce delay in causal mode')
    parser.add_argument('--causal', dest='causal', action='store_true', help='Run the downsampling in real-time (using delay)')
    parser.add_argument('--noncausal', dest='causal', action='store_false', help='Run the downsampling offline (output length may not be equal to input length)')

    parser.set_defaults(
        a='0,1,2,3,5,7',
        b='1,2,3,5,7',
        fir=True,
        fir_ntaps=128,
        fir_tol=0.01,
        iir_order=8,
        max_cascade=1,
        block_size=256,
        causal=False)

    args = parser.parse_args()
    print args
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
    fsn = fso * reduce_rational_list(rational_list)
    print 'Closest rational list: {}, fsn of {}'.format(rational_list, fsn)

    def create_fir(cutoff, ntaps, tolerance):
        b = remez(ntaps, [0.0, cutoff, cutoff + tolerance, 0.5], [1, 0])
        return b, np.ones(1, dtype=np.float64)

    def create_iir(cutoff, order):
        return iirfilter(order, cutoff * 2, btype='lowpass', ftype='butter')

    def create_identity():
        return (np.array([1.0], dtype=np.float64), np.array([1.0], dtype=np.float64))

    # Perform cascaded downsampling.
    wav_down = wav_f
    for upsample, downsample in rational_list:
        # Create upsampling interpolation filter.
        if upsample > 1:
            cutoff = 0.5 / float(upsample)
            if args.fir:
                interp_m = create_fir(cutoff, args.fir_ntaps, args.fir_tol)
            else:
                interp_m = create_iir(cutoff, args.iir_order)
        else:
            interp_m = create_identity()

        # Create reupsampling interpolation filter.
        if downsample > 1:
            cutoff = 0.5 / float(downsample)
            if args.fir:
                interp_l = create_fir(cutoff, args.fir_ntaps, args.fir_tol)
            else:
                interp_l = create_iir(cutoff, args.iir_order)
        else:
            interp_l = create_identity()

        wav_down = downsample_rt(wav_down, upsample, downsample, interp_m, interp_l, block_size=args.block_size, causal=args.causal)

    # Write file out.
    wav_out = (wav_down * 32767.0).astype(np.int16)
    wavwrite(args.out_fp, fso, wav_out)