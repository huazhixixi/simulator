# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:54:11 2019

@author: shang
"""
import resampy

"""
Created on Sat Nov 23 19:33:49 2019

@author: huazhilun
"""
import numpy as np
import numba
from numba import prange
from scipy.signal import lfilter
from dsp_numba_core import lms_numba_iter_core, decision


def __segment_axis(a, length, overlap, mode='cut', append_to_end=0):
    """
        Generate a new array that chops the given array along the given axis into
        overlapping frames.

        example:
        >>> segment_axis(arange(10), 4, 2)
        array([[0, 1, 2, 3],
               [2, 3, 4, 5],
               [4, 5, 6, 7],
               [6, 7, 8, 9]])

        arguments:
        a       The array to segment must be 1d-array
        length  The length of each frame
        overlap The number of array elements by which the frames should overlap

        end     What to do with the last frame, if the array is not evenly
                divisible into pieces. Options are:

                'cut'   Simply discard the extra values
                'pad'   Pad with a constant value

        append_to_end:    The value to use for end='pad'

        a new array will be returned.

    """

    if a.ndim != 1:
        raise Exception("Error, input array must be 1d")
    if overlap > length:
        raise Exception("overlap cannot exceed the whole length")

    stride = length - overlap
    row = 1
    total_number = length
    while True:
        total_number = total_number + stride
        if total_number > len(a):
            break
        else:
            row = row + 1

    # 一共要分成row行
    if total_number > len(a):
        if mode == 'cut':
            b = np.zeros((row, length), dtype=np.complex128)
            is_append_to_end = False
        else:
            b = np.zeros((row + 1, length), dtype=np.complex128)
            is_append_to_end = True
    else:
        b = np.zeros((row, length), dtype=np.complex128)
        is_append_to_end = False

    index = 0
    for i in range(row):
        b[i, :] = a[index:index + length]
        index = index + stride

    if is_append_to_end:
        last = a[index:]

        b[row, 0:len(last)] = last
        b[row, len(last):] = append_to_end

    return b


#######################################################################################################################
def upsample(symbol_x, sps):
    '''

    :param symbol_x: ndarray
    :param sps: sample per symbol
    :return: 2-d array after inserting zeroes between symbols
    '''

    symbol_x = np.atleast_2d(symbol_x)
    res = np.empty((symbol_x.shape[0], symbol_x.shape[1] * sps), dtype=symbol_x.dtype)
    for index, row in enumerate(symbol_x):
        row.shape = -1, 1
        row = np.tile(row, (1, sps))
        row[:, 1:] = 0
        row.shape = 1, -1
        res[index] = row
    return res


#######################################################################################################################


#########################################filter design##################################################################

def rrcfilter(alpha, span, sps):
    assert divmod(span * sps, 2)[1] == 0

    return _rcosdesign(span * sps, alpha, 1, sps)


def _rcosdesign(N, alpha, Ts, Fs):
    """
    Generates a root raised cosine (RRC) filter (FIR) impulse response.
    Parameters
    ----------
    N : int
        Length of the filter in samples.
    alpha : float
        Roll off factor (Valid values are [0, 1]).
    Ts : float
        Symbol period in seconds.
    Fs : float
        Sampling Rate in Hz.
    Returns
    ---------
    time_idx : 1-D ndarray of floats
        Array containing the time indices, in seconds, for
        the impulse response.
    h_rrc : 1-D ndarray of floats
        Impulse response of the root raised cosine filter.
    """

    T_delta = 1 / float(Fs)

    sample_num = np.arange(N + 1)
    h_rrc = np.zeros(N + 1, dtype=float)

    for x in sample_num:
        t = (x - N / 2) * T_delta
        if t == 0.0:
            h_rrc[x] = 1.0 - alpha + (4 * alpha / np.pi)
        elif alpha != 0 and t == Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) *
                                                (np.sin(np.pi / (4 * alpha)))) + (
                                                       (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        elif alpha != 0 and t == -Ts / (4 * alpha):
            h_rrc[x] = (alpha / np.sqrt(2)) * (((1 + 2 / np.pi) *
                                                (np.sin(np.pi / (4 * alpha)))) + (
                                                       (1 - 2 / np.pi) * (np.cos(np.pi / (4 * alpha)))))
        else:
            h_rrc[x] = (np.sin(np.pi * t * (1 - alpha) / Ts) +
                        4 * alpha * (t / Ts) * np.cos(np.pi * t * (1 + alpha) / Ts)) / \
                       (np.pi * t * (1 - (4 * alpha * t / Ts)
                                     * (4 * alpha * t / Ts)) / Ts)

    return h_rrc / np.sqrt(np.sum(h_rrc * h_rrc))


#########################################end###########################################################################

#######################################receiver DSP###################################################################
from scipy.signal import fftconvolve
from scipy.fftpack import fftfreq
import copy


class MatchedFilter(object):

    def __init__(self, roll_off, sps, span=1024):
        self.h = rrcfilter(roll_off, span, sps)
        self.delay = int(span / 2 * sps)
        # self.h = np.atleast_2d(self.h)

    def match_filter(self, signal):
        x = signal[0, :]
        y = signal[1, :]
        x = fftconvolve(x, self.h)
        y = fftconvolve(y, self.h)

        x = np.roll(x, -self.delay)
        y = np.roll(y, -self.delay)
        x = x[:signal.shape[1]]
        y = y[:signal.shape[1]]
        return x, y

    def inplace_match_filter(self, signal):
        x, y = self.match_filter(signal)
        signal[:] = np.array([x, y])
        return signal

    def __call__(self, signal):
        self.inplace_match_filter(signal)

        return signal


def cd_compensation(signal, spans, inplace=False):
    '''

    This function is used for chromatic dispersion compensation in frequency domain.
    The signal is Signal object, and a new sample is created from property data_sample_in_fiber

    :param signal: Signal object
    :param spans: Span object, the signal's has propagated through these spans
    :param inplace: if True, the compensated sample will replace the original sample in signal,or new ndarray will be r
    eturned

    :return: if inplace is True, the signal object will be returned; if false the ndarray will be returned
    '''
    try:
        import cupy as np
    except Exception:
        import numpy as np

    center_wavelength = signal.center_wavelength
    freq_vector = fftfreq(signal[0, :].shape[0], 1 / signal.fs_in_fiber)
    omeg_vector = 2 * np.pi * freq_vector

    sample = np.array(signal[:])

    if not isinstance(spans, list):
        spans = [spans]

    for span in spans:
        beta2 = -span.beta2(center_wavelength)
        dispersion = (-1j / 2) * beta2 * omeg_vector ** 2 * span.length
        dispersion = np.array(dispersion)
        for pol_index in range(sample.shape[0]):
            sample[pol_index, :] = np.fft.ifft(np.fft.fft(sample[pol_index, :]) * np.exp(dispersion))

    if inplace:
        if hasattr(np, 'asnumpy'):
            sample = np.asnumpy(sample)
        signal[:] = sample
        return signal
    else:
        if hasattr(np, 'asnumpy'):
            sample = np.asnumpy(sample)
        signal = copy.deepcopy(signal)
        signal[:] = sample
        return signal


def orthonormalize_signal(E, os=1):
    """
    Orthogonalizing signal using the Gram-Schmidt process _[1].
    Parameters
    ----------
    E : array_like
       input signal
    os : int, optional
        oversampling ratio of the signal
    Returns
    -------
    E_out : array_like
        orthonormalized signal
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process for more
       detailed description.
    """

    E = np.atleast_2d(E)
    E_out = np.empty_like(E)
    for l in range(E.shape[0]):
        # Center
        real_out = E[l, :].real - E[l, :].real.mean()
        tmp_imag = E[l, :].imag - E[l, :].imag.mean()

        # Calculate scalar products
        mean_pow_inphase = np.mean(real_out ** 2)
        mean_pow_quadphase = np.mean(tmp_imag ** 2)
        mean_pow_imb = np.mean(real_out * tmp_imag)

        # Output, Imag orthogonal to Real part of signal
        sig_out = real_out / np.sqrt(mean_pow_inphase) + \
                  1j * (tmp_imag - mean_pow_imb * real_out / mean_pow_inphase) / np.sqrt(mean_pow_quadphase)
        # Final total normalization to ensure IQ-power equals 1
        E_out[l, :] = sig_out - np.mean(sig_out[::os])
        E_out[l, :] = E_out[l, :] / np.sqrt(np.mean(np.abs(E_out[l, ::os]) ** 2))

    return E_out


def LMS(ntaps: int, samples_in: np.ndarray, mu: np.ndarray, training_sequence: np.ndarray = None,
        sps: int = 2, nloop: int = 1, training_time: int = 1, constl: np.ndarray = None,
        ):
    def init_tap_value(ntaps):
        hxx = np.zeros((1, ntaps), dtype=np.complex128)

        hxy = np.zeros((1, ntaps), dtype=np.complex128)
        hyx = np.zeros((1, ntaps), dtype=np.complex128)
        hyy = np.zeros((1, ntaps), dtype=np.complex128)

        hxx[0, ntaps // 2] = 1
        hyy[0, ntaps // 2] = 1
        return hxx, hxy, hyy, hyx

    mu = np.atleast_2d(mu)
    mu = mu[0]
    samples_in = np.atleast_2d(samples_in)
    if training_sequence is not None:
        training_sequence = np.atleast_2d(training_sequence)
        assert training_sequence.shape[0] == 2
        xtrain = training_sequence[0, ntaps // 2 // sps:]
        ytrain = training_sequence[1, ntaps // 2 // sps:]

    else:
        xtrain = None
        ytrain = None

    assert constl is not None
    assert samples_in.shape[0] == 2
    constl = np.atleast_2d(constl)

    xin = __segment_axis(samples_in[0], length=ntaps, overlap=ntaps - sps)
    yin = __segment_axis(samples_in[1], length=ntaps, overlap=ntaps - sps)
    wxx, wxy, wyy, wyx = init_tap_value(ntaps)

    res, error_xpol, error_ypol, wxx, wxy, wyy, wyx = lms_numba_iter_core(xin, yin, xtrain, ytrain, wxx, wxy, wyy, wyx,
                                                                          nloop, training_time, mu, constl)

    return res, error_xpol, error_ypol, wxx, wxy, wyy, wyx


def rotate_overall_phase(array, train_symbol, sps):
    array = np.atleast_2d(array)
    train_symbol = resampy.resample(train_symbol, 1, sps)
    phase = np.angle(np.mean(array / train_symbol, axis=1, keepdims=True))
    array = array * np.exp(-1j * phase)
    return array


def remove_dc(samples):
    samples = np.atleast_2d(samples)
    samples = samples - np.mean(samples, axis=1)
    return samples


def carrier_frequency_offset_dp(x, y, fs, fft_length):
    x = np.atleast_2d(x)
    x = x[0]
    y = np.atleast_2d(y)
    y = y[0]

    x_fft = np.fft.fft(x[-1 - fft_length:] ** 4)
    y_fft = np.fft.fft(y[-1 - fft_length:] ** 4)
    spectrum_x = np.fft.fftshift(np.abs(x_fft ** 2))
    spectrum_y = np.fft.fftshift(np.abs(y_fft ** 2))
    spectrum = spectrum_x + spectrum_y
    frequency = np.arange(-fs / 2 / 4, fs / 2 / 4, fs / fft_length / 4)
    index = np.argmax(spectrum)
    offset = frequency[index]
    return offset


def sync_correlation(symbol_tx, sample_rx, sps=1):
    '''

        :param symbol_tx: 发送符号 2d-array
        :param sample_rx: 接收符号，会相对于发送符号而言存在滞后 2d-array
        :param sps: samples per symbol
        :return: 收端符号移位之后的结果

        # 不会改变原信号

    '''
    from scipy.signal import correlate
    symbol_tx = np.atleast_2d(symbol_tx)
    sample_rx = np.atleast_2d(sample_rx)
    out = np.zeros_like(sample_rx)
    # assert sample_rx.ndim == 1
    # assert symbol_tx.ndim == 1
    assert sample_rx.shape[1] >= symbol_tx.shape[1]
    for i in range(symbol_tx.shape[0]):
        symbol_tx_temp = symbol_tx[i, :]
        sample_rx_temp = sample_rx[i, :]

        res = correlate(sample_rx_temp[::sps], symbol_tx_temp)

        index = np.argmax(np.abs(res))

        out[i] = np.roll(sample_rx_temp, sps * (-index - 1 + symbol_tx_temp.shape[0]))
    return out


def sync_with_pilot(seed, constl, ts_symbol_length, sample, sps):
    from resampy import resample
    np.random.seed(seed)
    constl = np.atleast_2d(constl)
    constl = constl[0]
    sample = np.atleast_2d(sample)

    index = np.random.randint(0, len(constl) - 1, (1, ts_symbol_length))
    pilot = np.empty((sample.shape[0], ts_symbol_length * sps))

    for i in pilot.shape[0]:
        pilot[i] = resample(constl[index], 1, sps)
    sample_after_sync = sync_correlation(pilot, sample, sps)
    return sample_after_sync


def superscalar(symbol_in, training_symbol, block_length, pilot_number, constl, g, filter_n=20):
    # delete symbols to assure the symbol can be divided into adjecent channels
    symbol_in = np.atleast_2d(symbol_in)
    training_symbol = np.atleast_2d(training_symbol)
    constl = np.atleast_2d(constl)
    assert symbol_in.shape[0] == 1
    assert training_symbol.shape[0] == 1
    assert constl.shape[0] == 1
    divided_symbols, divided_training_symbols = __initialzie_superscalar(symbol_in, training_symbol, block_length)
    angle = __initialzie_pll(divided_symbols, divided_training_symbols, pilot_number)

    divided_symbols = divided_symbols * np.exp(-1j * angle)
    divided_symbols = first_order_pll(divided_symbols, (constl), g)
    divided_symbols[0::2, :] = divided_symbols[0::2, ::-1]
    divided_symbols = divided_symbols.reshape((1, -1))
    # ml
    decision_symbols = np.zeros(divided_symbols.shape[1], dtype=np.complex)
    exp_decision(divided_symbols[0, :], constl[0, :], decision_symbols)

    hphase_ml = symbol_in[0, :len(decision_symbols)] / decision_symbols
    hphase_ml = np.atleast_2d(hphase_ml)
    h = np.ones((1, 2 * filter_n + 1))
    hphase_ml = lfilter(h[0, :], 1, hphase_ml)
    hphase_ml = np.roll(hphase_ml, -filter_n, axis=1)
    phase_ml = np.angle(hphase_ml)
    divided_symbols = symbol_in[:, :len(decision_symbols)] * np.exp(-1j * phase_ml)
    # ml completed
    divided_training_symbols[0::2, :] = divided_training_symbols[0::2, ::-1]
    divided_training_symbols = divided_training_symbols.reshape((1, -1))
    # scatterplot(divided_symbols,False,'pyqt')

    # filrst order pll

    return divided_symbols, divided_training_symbols


@numba.jit(nopython=True, parallel=True)
def first_order_pll(divided_symbols, constl, g):
    constl = np.atleast_2d(constl)
    phase = np.zeros((divided_symbols.shape[0], divided_symbols.shape[1]))
    for i in prange(divided_symbols.shape[0]):
        signal = divided_symbols[i, :]
        each_error = phase[i, :]
        for point_index, point in enumerate(signal):
            if point_index == 0:
                point = point * np.exp(-1j * 0)
            else:
                point = point * np.exp(-1j * each_error[point_index - 1])
            point_decision = decision(point, constl)
            signal[point_index] = point
            point_decision_conj = np.conj(point_decision)
            angle_difference = np.angle(point * point_decision_conj)

            if point_index > 0:
                each_error[point_index] = angle_difference * g + each_error[point_index - 1]
            else:
                each_error[point_index] = angle_difference * g

    return divided_symbols


def __initialzie_pll(divided_symbols, divided_training_symbols, pilot_number):
    '''
        There are pilot_number symbols of each row,the two adjecnt channel use the same phase,because they are simillar

    '''
    # get pilot symbol
    pilot_signal = divided_symbols[:, : pilot_number]
    pilot_traing_symbol = divided_training_symbols[:, :pilot_number]

    angle = (pilot_signal / pilot_traing_symbol)
    angle = angle.flatten()
    angle = angle.reshape(-1, 2 * pilot_number)
    angle = np.sum(angle, axis=1, keepdims=True)
    angle = np.angle(angle)

    angle_temp = np.zeros((angle.shape[0] * 2, angle.shape[1]), dtype=np.float)
    angle_temp[0::2, :] = angle
    angle_temp[1::2, :] = angle_temp[0::2, :]
    return angle_temp


def __initialzie_superscalar(symbol_in, training_symbol, block_length):
    # delete symbols to assure the symbol can be divided into adjecent channels
    symbol_in = np.atleast_2d(symbol_in)
    training_symbol = np.atleast_2d(training_symbol)
    assert symbol_in.shape[0] == 1
    symbol_length = len(symbol_in[0, :])
    assert divmod(block_length, 2)[1] == 0

    if divmod(symbol_length, 2)[1] != 0:
        # temp_symbol = np.zeros((symbol_in.shape[0], symbol_in.shape[1] - 1), dtype=np.complex)
        # temp_training_symbol = np.zeros((training_symbol.shape[0], training_symbol.shape[1] - 1), dtype=np.complex)
        temp_symbol = symbol_in[:, :-1]
        temp_training_symbol = training_symbol[:, :-1]
    else:
        temp_symbol = symbol_in
        temp_training_symbol = training_symbol

    # divide into channels
    channel_number = int(len(temp_symbol[0, :]) / block_length)
    if divmod(channel_number, 2)[1] == 1:
        channel_number = channel_number - 1
    divided_symbols = np.zeros((channel_number, block_length), dtype=np.complex)
    divided_training_symbols = np.zeros((channel_number, block_length), dtype=np.complex)
    for cnt in range(channel_number):
        divided_symbols[cnt, :] = temp_symbol[0, cnt * block_length:(cnt + 1) * block_length]
        divided_training_symbols[cnt, :] = temp_training_symbol[0, cnt * block_length:(cnt + 1) * block_length]
        if divmod(cnt, 2)[1] == 0:
            divided_symbols[cnt, :] = divided_symbols[cnt, ::-1]
            divided_training_symbols[cnt, :] = divided_training_symbols[cnt, ::-1]
    #             print(divided_training_symbols.shape)
    # First Order PLL

    return divided_symbols, divided_training_symbols


@numba.guvectorize([(numba.types.complex128[:], numba.types.complex128[:], numba.types.complex128[:])], '(n),(m)->(n)',
                   nopython=True)
def exp_decision(symbol, const, res):
    for index, sym in enumerate(symbol):
        # distance = sym-const
        distance = np.abs(sym - const)
        res[index] = const[np.argmin(distance)]
