# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 14:37:24 2019

@author: shang
"""

import matplotlib.pyplot as plt
import numpy as np

def scatterplot(array):
    array = np.atleast_2d(array)
    fig,axes = plt.subplots(1,array.shape[0],figsize=(15,5))

    if array.shape[0]>1:
        for index,row in enumerate(array):

            axes[index].scatter(row.real,row.imag)

    else:
        axes.scatter(array[0].real,array[0].imag)
    plt.tight_layout()
    plt.show()
    
#scatterplot(np.array([[1,2,3,4,5],[1,2,3,4,5]]))

from channel import LinearFiber
from optins import Edfa
from optins import WSS
from typing import List
def visualization_link(ins:List):
    for index,ele in enumerate(ins):
        if isinstance(ele,LinearFiber):
            if index == len(ins)-1:
                print('Fiber',end='')
            else:
                print('Fiber',end='->')
        if isinstance(ele,Edfa):
            if index == len(ins)-1:
                print('edfa',end='')
            else:
                print("EDFA",end='->')
        if isinstance(ele,WSS):
            if index == len(ins)-1:
                print('WSS','')
            else:
                print('wss',end='->')
    print('\n')

    
