import os

import numpy as np

from isnobal import varlistManager
from isnobal.isnobalconst import *

class RCParams:
    def __init__(self):
        self._d = {
                      IPW_IN: {},
                      IPW_PPT: {},
                      IPW_EM: {},
                      IPW_SNOW: {}
                    }

        for grp, band in varlistManager:
            self._d[grp][band] = {}

    def __getitem__(self, path):

        grp, _band = os.path.split(path)
        if grp == '':
            grp = _band
            return self._d[grp]

        try:
            _band, parameter = _band.split('.')
            return self._d[grp][_band][parameter]
        except: 
            return self._d[grp][_band]
        
    def __setitem__(self, path, value):
        grp, _band = os.path.split(path)
        _band, parameter = _band.split('.')
        self._d[grp][_band][parameter] = value

    def __str__(self):
        return str(self._d)
    
rcParams = RCParams()

# S_n
rcParams['in/S_n.colormap'] = 'Greys_r'
rcParams['in/S_n.ymin'] = 0
rcParams['in/S_n.ymax'] = np.log10(1201)
rcParams['in/S_n.transform'] = lambda x: np.log10(x + 1.0)

# I_lw
rcParams['in/I_lw.colormap'] = 'cool'
rcParams['in/I_lw.ymin'] = -4
rcParams['in/I_lw.ymax'] =  4
#rcParams['in/I_lw.transform'] = lambda x: (x + mu) / sigma


