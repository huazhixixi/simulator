from dataclasses import dataclass

import numpy as np
try:
    import torch
except Exception as e:
    pass
from scipy.constants import c

@dataclass
class FiberParam(object):
    alpha: float = 0.2
    D: float = 16.7
    gamma: float = 1.3
    length: float = 80
    step_length: float = 20 / 1000
    reference_wavelength: float = 1550
    slope: float = 0

    @property
    def alphalin(self):
        alphalin = self.alpha / (10 * np.log10(np.exp(1)))
        return alphalin

    @property
    def beta3_reference(self):
        res = (self.reference_wavelength * 1e-12 / 2 / np.pi / c / 1e-3) ** 2 * (
                2 * self.reference_wavelength * 1e-12 * self.D + (
                self.reference_wavelength * 1e-12) ** 2 * self.slope * 1e12)

        return res

    @property
    def beta2_reference(self):
        return -self.D * (self.reference_wavelength * 1e-12) ** 2 / 2 / np.pi / c / 1e-3

    def beta2(self, wave_length):
        '''

        :param wave_length: [m]
        :return: beta2 at wave_length [s^2/km]
        '''
        dw = 2 * np.pi * c * (1 / wave_length - 1 / (self.reference_wavelength * 1e-9))
        return self.beta2_reference + self.beta3_reference * dw

    def leff(self, length):
        '''

        :param length: the length of a fiber [km]
        :return: the effective length [km]
        '''
        effective_length = 1 - np.exp(-self.alphalin * length)
        effective_length = effective_length / self.alphalin
        return effective_length

def linear_prop(param:FiberParam,samples:np.ndarray,fs:float,center_wavelength:float,device:str):
    from scipy.fftpack import fft,ifft
    length = param.length
    xpol = samples[0]
    ypol = samples[1]
    freq = np.fft.fftfreq(len(xpol), 1 / fs)
    omeg = 2 * np.pi * freq
    D = -1 / 2 * param.beta2(center_wavelength) * omeg ** 2
    atten = -param.alphalin / 2
    
 
    frequency_domain_xpol = fft(xpol)
    frequency_domain_ypol = fft(ypol)
    frequency_domain_xpol = frequency_domain_xpol * np.exp(1j* D * length)
    frequency_domain_ypol = frequency_domain_ypol * np.exp(1j * D * length)

    xpol_time_domain = ifft(frequency_domain_xpol)
    ypol_time_domain = ifft(frequency_domain_ypol)
    xpol_time_domain = xpol_time_domain * np.exp(atten * length)
    ypol_time_domain = ypol_time_domain * np.exp(atten * length)
    
    samples = np.vstack((xpol_time_domain,ypol_time_domain))
    return samples
 
def fiber_cupy(param:FiberParam,samples:np.ndarray,fs:float,center_wavelength:float,device:str):
    import cupy as np
    from cupyx.scipy.fftpack import fft as improved_fft
    from cupyx.scipy.fftpack import ifft as improved_ifft
    from cupyx.scipy.fftpack import get_fft_plan
    samples = np.array(samples)
    plan = get_fft_plan(samples[0])
 
    def step(samples,step_length,step_eff):
        xpol = samples[0]
        ypol = samples[1]
        xpol,ypol = linear_prop(xpol,ypol,step_length/2)
        xpol,ypol = nonlinear_prop(xpol,ypol,step_eff)
        xpol,ypol = atteunation(xpol,ypol,step_length)
        xpol,ypol = linear_prop(xpol,ypol,step_length/2)
        return np.vstack((xpol,ypol))

    def atteunation(xpol,ypol,step_length):
        xpol = xpol * np.exp(atten * step_length)
        ypol = ypol * np.exp(atten * step_length)
        return xpol,ypol
        
    def nonlinear_prop(xpol,ypol,step_length):
        amplitude_xpol = (xpol[:, 0] ** 2 + xpol[:, 1] ** 2)
        amplitude_ypol = (ypol[:, 0] ** 2 + ypol[:, 1] ** 2)
        phase_rotation_xpol = 8 / 9 * (amplitude_xpol  + amplitude_ypol ) * gamma * step_length
        phase_rotation_ypol = phase_rotation_xpol
        
        xpol = xpol * np.exp(1j*2*np.pi*phase_rotation_xpol)
        ypol = ypol * np.exp(1j*2*np.pi*phase_rotation_ypol)
        
        return xpol,ypol
    
       

    def linear_prop(xpol,ypol,length):
        frequency_domain_xpol = improved_fft(xpol,overwrite_x=True,plan=plan)
        frequency_domain_ypol = improved_fft(ypol,overwrite_x=True,plan=plan)
        frequency_domain_xpol = frequency_domain_xpol * np.exp(1j* D * length)
        frequency_domain_ypol = frequency_domain_ypol * np.exp(1j * D * length)

        xpol_time_domain = improved_ifft(frequency_domain_xpol,overwrite_x=True,plan=plan)
        ypol_time_domain = improved_ifft(frequency_domain_ypol,overwrite_x=True,plan=plan)

        return xpol_time_domain, ypol_time_domain
        
    
    gamma = param.gamma
    length = param.length
    step_length = param.step_length
    xpol = samples[0]
    freq = np.fft.fftfreq(len(xpol), 1 / fs)
    omeg = 2 * np.pi * freq
    D = -1 / 2 * param.beta2(center_wavelength) * omeg ** 2
    atten = -param.alphalin / 2
    
    step_number = length / step_length
    step_number = int(np.floor(step_number))

    last_length = length - step_number * step_length
    step_eff = 1 - np.exp(-param.alphalin * step_length)
    step_eff = step_eff / param.alphalin
    last_length_eff = 1 - np.exp(-param.alphalin * last_length)
    last_length_eff = last_length_eff / param.alphalin
    
    for step_index in range(step_number):
        samples = step(samples,step_length,step_eff)
    
    if last_length_eff:
        samples = step(samples,last_length,last_length_eff)
    
    return samples

def fiber_prop_torch(param: FiberParam, samples: np.ndarray, fs: float, center_wavelength: float, device: torch.device):

    def step(xpol, ypol, step_length, step_eff):
        xpol, ypol = dispersion(xpol, ypol, step_length / 2)

        xpol_phase, ypol_phase = nonlinear_rotation(xpol, ypol, step_eff)
        xpol = complex_exp(xpol, xpol_phase)
        ypol = complex_exp(ypol, ypol_phase)

        xpol, ypol = dispersion(xpol, ypol, step_length / 2)

        xpol = xpol * torch.exp(atten * step_length)
        ypol = ypol * torch.exp(atten * step_length)
        return xpol, ypol

    def nonlinear_rotation(xpol, ypol, step_length):
        amplitude_xpol = (xpol[:, 0] ** 2 + xpol[:, 1] ** 2)
        amplitude_ypol = (ypol[:, 0] ** 2 + ypol[:, 1] ** 2)
        phase_rotation_xpol = 8 / 9 * (amplitude_xpol  + amplitude_ypol ) * gamma * step_length
        phase_rotation_ypol = phase_rotation_xpol
        return phase_rotation_xpol, phase_rotation_ypol

    def complex_exp(op1, exp_pol):
        real = op1[:, 0]
        imag = op1[:, 1]
        real_temp = real * torch.cos(exp_pol) - imag * torch.sin(exp_pol)
        imag_temp = real * torch.sin(exp_pol) + imag * torch.cos(exp_pol)

        op1[:, 0] = real_temp
        op1[:, 1] = imag_temp
        return op1

    def dispersion(xpol, ypol, length):
        frequency_domain_xpol = torch.fft(xpol, 1)
        frequency_domain_ypol = torch.fft(ypol, 1)
        frequency_domain_xpol = complex_exp(frequency_domain_xpol, D * length)
        frequency_domain_ypol = complex_exp(frequency_domain_ypol, D * length)

        xpol_time_domain = torch.ifft(frequency_domain_xpol, 1)
        ypol_time_domain = torch.ifft(frequency_domain_ypol, 1)

        return xpol_time_domain, ypol_time_domain

    gamma = param.gamma
    length = param.length
    step_length = param.step_length

    xpol = samples[0]
    freq = np.fft.fftfreq(len(xpol), 1 / fs)
    omeg = 2 * np.pi * freq
    omeg = torch.tensor(omeg).double().to(device)
    xpol_real = xpol.real[:, np.newaxis]
    xpol_imag = xpol.imag[:, np.newaxis]
    ypol = samples[1]
    ypol_real = ypol.real[:, np.newaxis]
    ypol_imag = ypol.imag[:, np.newaxis]

    xpol = np.hstack((xpol_real, xpol_imag))
    xpol = torch.tensor(xpol).to(device)

    ypol = np.hstack((ypol_real, ypol_imag))
    ypol = torch.tensor(ypol).double().to(device)
    D = -1 / 2 * param.beta2(center_wavelength) * omeg ** 2

    atten = -param.alphalin / 2
    atten = torch.tensor(atten).double().to(device)
    step_number = length / step_length
    step_number = int(np.floor(step_number))

    last_length = length - step_number * step_length
    step_eff = 1 - np.exp(-param.alphalin * step_length)
    step_eff = step_eff / param.alphalin
    last_length_eff = 1 - np.exp(-param.alphalin * last_length)
    last_length_eff = last_length_eff / param.alphalin
    import tqdm
    for step_ith in tqdm.tqdm(range(step_number)):
        xpol, ypol = step(xpol, ypol, step_length, step_eff)

    if last_length:
        xpol, ypol = step(xpol, ypol, last_length, last_length_eff)
    try:
        xpol = xpol.numpy()
        ypol = ypol.numpy()
        
    except TypeError:
        xpol = xpol.cpu().numpy()
        ypol = ypol.cpu().numpy()
    xpol = xpol[:, 0] + xpol[:, 1] * 1j
    ypol = ypol[:, 0] + ypol[:, 1] * 1j

    xpol.shape = 1, -1
    ypol.shape = 1, -1

    samples = np.vstack((xpol, ypol))

    return samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots()
    axes.set_xlabel("xixi")
    print(axes.xaxis.get_ticklabels())
    plt.show()
