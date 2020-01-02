from simulation.SignalDefine import QamSignal
symbol_rate = 20e9
sps = 2
subcarrier_number = 4

subcarrier_spacing = symbol_rate/subcarrier_number *(1+0.1)

sps_in_fiber = sps * subcarrier_number
signals = []
from simulation.optins import Multiplex
for i in range(subcarrier_number):
    signals.append(QamSignal(sps,sps_in_fiber,{'roll_off':0.1},2**16,4,0
                             ,symbol_rate/subcarrier_number,seed=i,center_frequency=0+subcarrier_spacing*i))

for signal in signals:
    signal.prepare()

wdm_signal = Multiplex.mux_signal(signals)
# wdm_signal.spectrum_plot(2**16)

from resampy import resample
from scipy.io import savemat

dac_rate = 80e9
wdm_signal.samples = resample(wdm_signal.samples,sr_orig=symbol_rate*sps,sr_new=dac_rate,axis=1)
wdm_signal.fs_in_fiber = dac_rate
wdm_signal.spectrum_plot(2**16)

import joblib
joblib.dump(wdm_signal.symbol_each_channel,'tx_symbols')
savemat('samples_to_tx.mat',dict(samples = wdm_signal.samples))