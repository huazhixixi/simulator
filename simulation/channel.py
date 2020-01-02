from numpy.fft import fftfreq
from scipy.fftpack import fft, ifft

try:

    import cupy as cp
    from cupyx.scipy.fftpack import fft as improved_fft
    from cupyx.scipy.fftpack import ifft as improved_ifft
    from cupyx.scipy.fftpack import get_fft_plan
except Exception as e:
    pass


import numpy as np
from scipy.constants import c
import math





class LinearFiber(object):
    '''
        property:
            self.alpha  [db/km]
            self.D [ps/nm/km]
            self.length [km]
            self.reference_wave_length:[nm]
            self.beta2: caculate beta2 from D,s^2/km
            self.slope: derivative of self.D ps/nm^2/km
            self.beta3_reference: s^3/km
        method:
            __call__: the signal will
    '''

    def __init__(self, alpha, D, length, slope=0, reference_wave_length=1550):
        self.alpha = alpha
        self.D = D
        self.length = length
        self.reference_wave_length = reference_wave_length
        self.slope = slope

    @property
    def beta3_reference(self):
        res = (self.reference_wave_length * 1e-12 / 2 / np.pi / c / 1e-3) ** 2 * (
                2 * self.reference_wave_length * 1e-12 * self.D + (
                self.reference_wave_length * 1e-12) ** 2 * self.slope * 1e12)

        return res

    @property
    def alpha_lin(self):
        # [1/km]
        return 0.1 * self.alpha * np.log(10)

    def leff(self, length):
        '''

        :param length: the length of a fiber [km]
        :return: the effective length [km]
        '''
        effective_length = 1 - np.exp(-self.alpha_lin * length)
        effective_length = effective_length / self.alpha_lin
        return effective_length

    @property
    def beta2_reference(self):
        return -self.D * (self.reference_wave_length * 1e-12) ** 2 / 2 / np.pi / c / 1e-3

    def beta2(self, wave_length):
        '''

        :param wave_length: [m]
        :return: beta2 at wave_length [s^2/km]
        '''
        dw = 2 * np.pi * c * (1 / wave_length - 1 / (self.reference_wave_length * 1e-9))
        return self.beta2_reference + self.beta3_reference * dw

    def _prop(self, signal):
        '''

        :param signal: signal object to propagation across this fiber
        :return: ndarray
        '''
        center_lambda = signal.center_wavelength

        after_prop = np.zeros_like(signal[:])
        for pol in range(0, int(signal.is_pol+1)):
            sample = signal[pol, :]
            sample_fft = fft(sample)
            freq = fftfreq(signal[:].shape[1], 1 / signal.fs_in_fiber)
            omeg = 2 * np.pi * freq

            after_prop[pol, :] = sample_fft * np.exp(-self.alpha_lin * self.length / 2)
            after_prop[pol, :] = ifft(after_prop[pol, :])

            disp = np.exp(-1j / 2 * self.beta2(center_lambda) * omeg ** 2 * self.length)
            after_prop[pol, :] = ifft(fft(after_prop[pol, :]) * disp)

        return np.atleast_2d(after_prop)

    def inplace_prop(self, signal):
        after_prop = self._prop(signal)

        for i in range(int(signal.is_pol+1)):
            signal[i, :] = after_prop[i, :]
        return signal

    def __call__(self, signal):
        self.inplace_prop(signal)
        return signal

    def __str__(self):
        string = f"alpha is {self.alpha} [db/km]\n" \
            f"beta2 is {self.beta2_reference} [s^2/km]\n" \
            f"beta3 is {self.beta3_reference} []\n" \
            f"D is {self.D} ps/nm/km\n" \
            f"length is {self.length} km\n" \
            f"reference wave length is {self.reference_wave_length * 1e9} [nm]"

        return string

    def __repr__(self):
        return self.__str__()


class NonlinearFiber(LinearFiber):

    def __init__(self, alpha, D, length, gamma, slope=0, step_length=5 / 1000, reference_wave_length=1550):
        super().__init__(alpha, D, length, slope=slope, reference_wave_length=reference_wave_length)
        self.gamma = gamma
        self.step_length = step_length


    @property
    def step_length_eff(self):
        return (1 - np.exp(-self.alpha_lin * self.step_length)) / self.alpha_lin

    def cupy_prop(self, signal):

        if not signal.is_pol :
            raise Exception("only dp signal supported at this time")
        step_number = self.length / self.step_length
        step_number = int(np.floor(step_number))
        temp = np.zeros_like(signal[:])
        freq = fftfreq(signal.shape[1], 1 / signal.fs_in_fiber)
        freq_gpu = cp.asarray(freq)
        omeg = 2 * np.pi * freq_gpu
        D = -1j / 2 * self.beta2(signal.center_wavelength) * omeg ** 2
        N = 8 / 9 * 1j * self.gamma
        atten = -self.alpha_lin / 2

        time_x = cp.asarray(signal[0, :])
        time_y = cp.asarray(signal[1, :])

        plan = get_fft_plan(time_x)

        for i in range(step_number):

            time_x, time_y = self.linear_prop_cupy(D, time_x, time_y, self.step_length / 2,plan)
            time_x, time_y = self.nonlinear_prop_cupy(N, time_x, time_y)
            time_x = time_x * math.exp(atten * self.step_length)
            time_y = time_y * math.exp(atten * self.step_length)


            time_x, time_y = self.linear_prop_cupy(D, time_x, time_y, self.step_length / 2,plan)

        last_step = self.length - self.step_length * step_number
        last_step_eff = (1 - np.exp(-self.alpha_lin * last_step)) / self.alpha_lin
        if last_step == 0:
            time_x = cp.asnumpy(time_x)
            time_y = cp.asnumpy(time_y)
            temp[0, :] = time_x
            temp[1, :] = time_y

            return temp
        else:

            time_x, time_y = self.linear_prop_cupy(D, time_x, time_y, last_step / 2,plan)
            time_x, time_y = self.nonlinear_prop_cupy(N, time_x, time_y, last_step_eff)
            time_x = time_x * math.exp(atten * last_step)
            time_y = time_y * math.exp(atten * last_step)
            time_x, time_y = self.linear_prop_cupy(D, time_x, time_y, last_step / 2,plan)

            temp[0, :] = cp.asnumpy(time_x)
            temp[1, :] = cp.asnumpy(time_y)

        return temp

    def nonlinear_prop_cupy(self, N, time_x, time_y, step_length=None):

        if step_length is None:
            time_x = time_x * cp.exp(
                N * self.step_length_eff * (cp.abs(time_x) ** 2 + cp.abs(
                    time_y) ** 2))
            time_y = time_y * cp.exp(
                N * self.step_length_eff * (cp.abs(time_x) ** 2 + cp.abs(time_y) ** 2))
        else:
            time_x = time_x * cp.exp(
                N * step_length * (cp.abs(time_x) ** 2 + cp.abs(
                    time_y) ** 2))
            time_y = time_y * cp.exp(
                N * step_length * (cp.abs(time_x) ** 2 + cp.abs(time_y) ** 2))

        return time_x, time_y

    def linear_prop_cupy(self, D, timex, timey, length,plan):

        freq_x = improved_fft(timex,overwrite_x=True,plan=plan)
        freq_y = improved_fft(timey, overwrite_x=True, plan=plan)

        freq_x = freq_x * cp.exp(D * length)
        freq_y = freq_y * cp.exp(D * length)

        time_x = improved_ifft(freq_x,overwrite_x=True,plan=plan)
        time_y = improved_ifft(freq_y,overwrite_x=True,plan=plan)
        return time_x, time_y


    def inplace_prop(self, signal):

        after_prop = self.cupy_prop(signal)

        signal[:] = np.array(after_prop)
        return signal

    def __call__(self, signal):
        self.inplace_prop(signal)
        return signal

    def __str__(self):
        string = super(NonlinearFiber, self).__str__()
        string = f"{string}" \
            f"the step length is {self.step_length} [km]\n"
        return string

    def __repr__(self):
        return self.__str__()