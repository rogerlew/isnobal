from __future__ import print_function

import math
import os

from pprint import pprint as pp

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

import numpy as np

from isnobal.isnobalconst import *
from isnobal.viz_lib.colormaps import *

if __name__ == "__main__":
    data = [[j for j in xrange(1001)] for i in xrange(100)]
    data = np.array(data, dtype=np.float32).T
    data /= 1000.0
    
    mpl.rc('font', family='serif')
    mpl.rc('font', serif='Arial') 
    
    # SWE legend
    fig = plt.figure(figsize=(2, 2))
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    set_foregroundcolor(ax, (1,1,1))
    
    plt.imshow(swe_cm(data), origin='lower')
    plt.xticks([])
    plt.yticks([0, 200, 400, 600, 800, 1000],
               ["0.000", ">0.010", "0.031", "0.125", "0.500", "2.000"])
    plt.text(50, 1100, 'm', ha='center', va='bottom', color='white')
    plt.savefig(r'D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL\SWE_legend.png', dpi=128, transparent=True)
    plt.close()

    
    # delta Q legend
    fig = plt.figure(figsize=(2, 2))
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    set_foregroundcolor(ax, (1,1,1))
    
    plt.imshow(delta_Q_cm(data), origin='lower')
    plt.xticks([])
    plt.yticks([0, 250, 500, 750, 1000],
               ["-700", "-26", "0", "26", "700"])
    plt.text(50, 1100, u'W / m\u00B2', ha='center', va='bottom', color='white')
    plt.savefig(r'D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL\delta_Q_legend.png', dpi=128, transparent=True)
    plt.close()

    
    # melt legend
    fig = plt.figure(figsize=(2, 2))
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    set_foregroundcolor(ax, (1,1,1))
    
    plt.imshow(melt_cm(data), origin='lower')
    plt.xticks([])
    plt.yticks([0, 200, 400, 600, 800, 1000],
               ["0.000", "0.031", "0.125", "0.500", "2.000", "8.000"])
    plt.text(50, 1100, u'kg', ha='center', va='bottom', color='white')
    plt.savefig(r'D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL\melt_legend.png', dpi=128, transparent=True)
    plt.close()
    
    # cummulative SWI legend
    fig = plt.figure(figsize=(2, 2))
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    set_foregroundcolor(ax, (1,1,1))
    
    plt.imshow(swi_cm(data), origin='lower')
    plt.xticks([])
    plt.yticks([0, 250, 500, 750, 1000],
               ["0", "30", "60", "90", "120"])
    plt.text(50, 1100, u'kg', ha='center', va='bottom', color='white')
    plt.savefig(r'D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL\cummSWI_legend.png', dpi=128, transparent=True)
    plt.close()
    
    # gradient weight legend
    fig = plt.figure(figsize=(2, 2))
    fig.subplots_adjust(top=0.8)
    ax = fig.add_subplot(111)
    ax.yaxis.tick_right()
    set_foregroundcolor(ax, (1,1,1))

    cm = plt.get_cmap("nipy_spectral")
    
    plt.imshow(cm(data), origin='lower')
    plt.xticks([])
    plt.yticks([0, 200, 400, 600, 800, 1000],
               ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
#    plt.text(50, 1100, u'kg', ha='center', va='bottom', color='white')
    plt.savefig(r'D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL\gradW_legend.png', dpi=128, transparent=True)
    plt.close()
    
