import numpy as np

from scipy.signal import lfilter

from util import read_from_ring_buffer, write_to_ring_buffer, calc_block_range, upsample_crude, downsample_crude

def multirate_rt(x, m, l, m_offset, filt, zfilt=None):
    assert m >= 0 and l > 0
    if m == 0:
        return np.zeros((0,), dtype=x.dtype)

    if zfilt is None:
        zfilt = np.zeros(max(filt[0].shape[0], filt[1].shape[0]) - 1, dtype=np.float64)

    # Upsample
    if m > 0:
        xup = upsample_crude(x, m)
    else:
        xup = np.zeros((0,), dtype=x.dtype)
    
    # Interpolate
    if xup.shape[0] > 0:
        xinterp, zfilt = lfilter(filt[0], filt[1], xup, zi=zfilt)
    else:
        xinterp = np.zeros_like(xup)

    # Apply gain
    xinterp *= float(m)
    
    # Decimate
    extra, xdown = downsample_crude(xinterp[m_offset:], l)
    m_offset = l - extra
    if m_offset == l:
        m_offset = 0

    return xdown, m_offset, zfilt

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
    parser.add_argument('--gain', type=float, help='Gain before casting to PCM')

    parser.set_defaults(
        a='0,1,2,3,5,7',
        b='1,2,3,5,7',
        fir=True,
        fir_ntaps=128,
        fir_tol=0.01,
        iir_order=8,
        max_cascade=1,
        block_size=256,
        causal=False,
        gain=1.0)

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

    # Create filters.
    filt_alias = []
    zfilt_alias = []
    filt_antialias = []
    zfilt_antialias = []
    for upsample, downsample in rational_list:
        # Create upsampling interpolation filter.
        if upsample > 1:
            cutoff = 0.5 / float(upsample)
            if args.fir:
                interp_alias = create_fir(cutoff, args.fir_ntaps, args.fir_tol)
            else:
                interp_alias = create_iir(cutoff, args.iir_order)
        else:
            interp_alias = create_identity()
        filt_alias.append(interp_alias)
        zfilt_alias.append(None)

        # Create reupsampling interpolation filter.
        larger = max(upsample, downsample)
        if downsample > 1:
            cutoff = 0.5 / float(larger)
            if args.fir:
                interp_antialias = create_fir(cutoff, args.fir_ntaps, args.fir_tol)
            else:
                interp_antialias = create_iir(cutoff, args.iir_order)
        else:
            interp_antialias = create_identity()
        filt_antialias.append(interp_antialias)
        zfilt_antialias.append(None)

    # Create delay buffers.
    _, block_max = calc_block_range(args.block_size, rational_list)
    delay_buffer = np.zeros(args.block_size * 2, dtype=np.float64)
    delay_read_idx = 0
    delay_write_idx = 0
    num_written = 0
    num_read = 0

    # Initialize offsets.
    alias_offsets = [0 for _ in xrange(len(rational_list))]
    antialias_offsets = [0 for _ in xrange(len(rational_list))]

    # Perform cascaded downsampling.
    wav_out = []
    for i in xrange(0, wav_f.shape[0], args.block_size):
        block_last_rate = wav_f[i:i + args.block_size]
        block_size = block_last_rate.shape[0]

        for i, (upsample, downsample) in enumerate(rational_list):
            block_last_rate, offset, zfilt = multirate_rt(block_last_rate, upsample, downsample, alias_offsets[i], filt_alias[i], zfilt_alias[i])
            alias_offsets[i] = offset
            zfilt_alias[i] = zfilt

        for i in reversed(xrange(len(rational_list))):
            upsample, downsample = rational_list[i]

            block_last_rate, offset, zfilt = multirate_rt(block_last_rate, downsample, upsample, antialias_offsets[i], filt_antialias[i], zfilt_antialias[i])
            antialias_offsets[i] = offset
            zfilt_antialias[i] = zfilt

        delay_write_idx = write_to_ring_buffer(delay_buffer, block_last_rate, delay_write_idx)
        num_written += block_last_rate.shape[0]
        delay_read_idx, block_last_rate_del = read_from_ring_buffer(delay_buffer, block_size, delay_read_idx)
        num_read += block_size

        assert num_written >= num_read

        wav_out.append(block_last_rate_del)
    wav_out = np.concatenate(wav_out)

    # Write file out.
    wav_out *= args.gain
    wav_out = (wav_out * 32767.0).astype(np.int16)
    wavwrite(args.out_fp, fso, wav_out)