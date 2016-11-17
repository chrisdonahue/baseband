import numpy as np

from scipy.signal import lfilter

def process_block(block, block_size):
    global upsample
    global downsample
    global zup
    global zdown
    global iir_up_b
    global iir_up_a
    global iir_down_b
    global iir_down_a
    global down_offset
    global redown_offset
    
    # Upsample
    xup = np.zeros((upsample, block_size))
    xup[0] = block
    xup = xup.T.flatten()
    
    # Interpolate
    xinterp, zup = lfilter(iir_up_b, iir_up_a, xup, zi=zup)
    xinterp_len = xinterp.shape[0]
    assert xinterp_len == block_size * upsample
    
    # Manage edge conditions for downsampling
    xedge = xinterp[down_offset:]
    extra = xedge.shape[0] % downsample
    if extra > 0:
        xedge = np.concatenate([xedge, np.zeros(downsample - extra, dtype=np.float64)])
    assert xedge.shape[0] % downsample == 0
    down_offset = (downsample - ((xinterp_len - down_offset) % downsample))
    if down_offset == downsample:
        down_offset = 0
    
    # Decimate
    xdown = np.reshape(xedge, (-1, downsample))[:, 0]
    
    # Reupsample
    xreup = np.zeros((downsample, xdown.shape[0]))
    xreup[0] = xdown
    xreup = xreup.T.flatten()
    
    # Reinterpolate
    xreinterp, zdown = lfilter(iir_down_b, iir_down_a, xreup, zi=zdown)
    xreinterp_len = xreinterp.shape[0]
    
    # Manage edge conditions for redownsampling
    xreedge = xreinterp[redown_offset:]
    extra = xreedge.shape[0] % upsample
    if extra > 0:
        xreedge = np.concatenate([xreedge, np.zeros(upsample - extra, dtype=np.float64)])
    assert xreedge.shape[0] % upsample == 0
    redown_offset = (upsample - ((xreinterp_len - redown_offset) % upsample))
    if redown_offset == upsample:
        redown_offset = 0
    
    # Redownsample
    xrecons = np.reshape(xreedge, (-1, upsample))[:, 0]
    
    return xrecons


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
    block_size = 1

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
    upsample, downsample = rationalize_real(fsn / fso, A=A, B=B)

    # Calc interpolation filters.
    iir_up_b, iir_up_a = iirfilter(order, 1.0 / upsample, btype='lowpass', ftype='butter')
    iir_down_b, iir_down_a = iirfilter(order, 1.0 / downsample, btype='lowpass', ftype='butter')

    # Calc initial filter states.
    zup = lfilter_zi(iir_up_b, iir_up_a)
    zdown = lfilter_zi(iir_down_b, iir_down_a)

    # Set initial downsampling offset states.
    down_offset = 0
    redown_offset = 0

    # Calc delay state.
    blockout_minlen, blockout_maxlen = calc_block_range(block_size, upsample, downsample)
    delay = block_size - blockout_minlen
    overflow = blockout_maxlen - block_size
    storage_len = delay + block_size + overflow
    storage = np.zeros(storage_len, dtype=np.float64)
    read_idx = 0
    write_idx = delay

    # Perform RT downsampling.
    wav_down = []
    for i in xrange(0, len(wav_f), block_size):
        block = wav_f[i:i + block_size]
        blockout = process_block(block, len(block))
        write_idx = write_to_ring_buffer(storage, blockout, write_idx)
        read_idx, blockdel = read_from_ring_buffer(storage, block.shape[0], read_idx)
        wav_down.append(blockdel)
    wav_down = np.concatenate(wav_down)

    assert wav_f.shape == wav_down.shape

    # Write file out.
    wav_out = (wav_down * 32767.0).astype(np.int16)
    wavwrite(out_fp, fso, wav_out)