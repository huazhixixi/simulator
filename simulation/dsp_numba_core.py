# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:55:22 2019

@author: shang
"""

import numba
import numpy as np


@numba.jit(cache=True)
def lms_numba_iter_core(xin,yin,xtrain,ytrain,wxx,wxy,wyy,wyx,nloop,training_time,mu,constl):
    error_xpol = np.empty((nloop,xin.shape[0]),dtype = np.complex128)
    error_ypol = np.empty((nloop,yin.shape[0]),dtype = np.complex128)

    for iter_index in range(nloop):

        for i in range(xin.shape[0]):

            xout = np.sum(xin[i,::-1]*wxx) + np.sum(yin[i,::-1]*wxy)
            yout = np.sum(yin[i,::-1]*wyy) + np.sum(xin[i,::-1]*wyx)


            if training_time:
                error_xpol[iter_index,i] = xtrain[i] - xout
                error_ypol[iter_index,i] = ytrain[i] - yout

            else:
                dd_xresults = __decision(xout,constl)
                dd_yresults = __decision(yout,constl)
                error_xpol[iter_index,i] = xtrain[i] - dd_xresults
                error_ypol[iter_index,i] = ytrain[i] - dd_yresults
            if training_time:
                mu_for_gradi_decend = mu[0]
            else:
                mu_for_gradi_decend = mu[1]

            wxx = wxx + mu_for_gradi_decend*error_xpol[iter_index,i]*np.conj(xin[i,::-1])
            wxy = wxy + mu_for_gradi_decend*error_xpol[iter_index,i]*np.conj(yin[i,::-1])
            wyx = wyx + mu_for_gradi_decend* error_ypol[iter_index,i] *np.conj(xin[i,::-1])
            wyy = wyy + mu_for_gradi_decend* error_ypol[iter_index,i] *np.conj(yin[i,::-1])
        training_time = training_time - 1

    xout = np.sum(xin[:, ::-1] * wxx,axis=1) + np.sum(yin[:, ::-1] * wxy,axis=1)
    yout = np.sum(yin[:, ::-1] * wyy,axis=1) + np.sum(xin[:, ::-1] * wyx,axis=1)
    # xout = xout.flatten()
    # yout = yout.flatten()

    restults = np.ones((2,len(xout)),dtype = np.complex128)
    restults[0] = xout
    restults[1] = yout


    return restults,error_xpol,error_ypol,wxx,wxy,wyy,wyx

@numba.njit(cache=True)
def __decision(x,constl):
    constl = np.atleast_2d(constl)
    distance = np.abs(x - constl[0])
    index = np.argmin(distance)
    return constl[0,index]

decision = __decision