import dataclasses
from typing import List

import numpy as np
import resampy
from collections import namedtuple
from dsp import rrcfilter
from dsp import upsample
from scipy.signal import fftconvolve
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Signal(object):
    def __init__(self,sps,sps_in_fiber,baudrate,center_frequency):
        print('baudrate must be in hz')

        self.sps = sps
        self.sps_in_fiber = sps_in_fiber
        self.baudrate = baudrate

        self.ds_in_fiber = None
        self.ds_in_dsp = None
        self.center_frequency = center_frequency

        self.msg = None
        self.symbol = None
    @property
    def fs_in_fiber(self):
        return self.sps_in_fiber * self.baudrate

    @property
    def fs(self):
        return self.sps * self.baudrate

    def __getitem__(self, slice):
        assert self.ds_in_fiber is not None
        return self.ds_in_fiber[slice]

    def __setitem__(self, slice, value):
        if self.ds_in_fiber is None:
            self.ds_in_fiber = value
        else:
            self.ds_in_fiber[slice] = value

    @property
    def center_wavelength(self):
        from scipy.constants import c
        return c/self.center_frequency

    @property
    def shape(self):
        return self[:].shape

class QamSignal(Signal):
    symbol_dict = {
        4: '4qam.npy',
        16: '16qam.npy'
    }

    def __init__(self, sps, sps_in_fiber, rrc_param, length, order, power, baudrate, seed=0, is_pol=1,center_frequency=193.1E12):
        super(QamSignal, self).__init__(sps,sps_in_fiber,baudrate,center_frequency)
        self.rrc_param = rrc_param
        self.order = order
        self.length = length
        self.seed = seed
        self.msg = None
        self.symbol = None
        self.is_pol = is_pol
        self.power = power
        self.init()

    def init(self):
        np.random.seed(self.seed)
        self.msg = np.random.randint(0, self.order, (self.is_pol + 1, self.length))
        constl = np.load(BASE_DIR+'/'+QamSignal.symbol_dict[self.order])
        constl = np.atleast_2d(constl)
        self.symbol = np.ones_like(self.msg, dtype=np.complex)
        for index, row in enumerate(self.msg):
            for i in range(self.order):
                mask = row == i
                self.symbol[index,mask] = constl[0, i]


    def prepare(self):
        roll_off = self.rrc_param.get('roll_off', 0.02)
        span = self.rrc_param.get('span', 1024)
        self.ds_in_dsp = upsample(self.symbol, self.sps)
        self.ds_in_dsp = self.pulse_shaping(roll_off, span)

        self.ds_in_fiber = resampy.resample(self.ds_in_dsp, self.sps, self.sps_in_fiber, axis=1)
        self.set_power()

    def set_power(self):
        factor = np.mean(np.abs(self.ds_in_fiber) ** 2, axis=1, keepdims=True)
        self.ds_in_fiber = self.ds_in_fiber / np.sqrt(factor)
        power_linear =( 10 ** (self.power / 10)) / 1000 / 2
        self.ds_in_fiber = np.sqrt(power_linear) * self.ds_in_fiber

    def power_meter(self):
        power = np.mean(np.abs(self.ds_in_fiber) ** 2,axis=1)
        total_linear_power = np.sum(power) * 1000
        total_linear_power_dbm = 10 * np.log10(total_linear_power)
        power = power * 1000
        power_dbm = 10 * np.log10(power)
        res = namedtuple('power','power_linear power_dbm total_power_linear total_power_linear_dbm')
        res = res(power,power_dbm,total_linear_power,total_linear_power_dbm)

        return res




    def pulse_shaping(self, roll_off, span):
        rrc_filter_tap = rrcfilter(roll_off, span, self.sps)
        delay = span / 2 * self.sps
        delay = int(delay)
        for index, row in enumerate(self.ds_in_dsp):
            row = fftconvolve(row, np.atleast_2d(rrc_filter_tap)[0])
            row = np.roll(row, -delay)
            row = row[:self.sps * self.symbol.shape[1]]
            self.ds_in_dsp[index] = row
        return self.ds_in_dsp

@dataclasses.dataclass
class WdmSignal(object):
    symbol_each_channel: List[np.ndarray]
    frequencies: List
    samples: np.ndarray
    fs_in_fiber:float


    def spectrum_plot(self,nfft_size):
        import matplotlib.pyplot as plt
        plt.psd(self.samples[0],NFFT=nfft_size,Fs=self.fs_in_fiber)
        plt.show()




