import os

import numpy as np

from isnobal import varlistManager
from isnobal.isnobalconst import *

class BandMetadata:
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
    
meta_db = BandMetadata()

# Inputs
meta_db['in/I_lw.Description'] = 'incoming thermal (long-wave) radiation'
meta_db['in/I_lw.Units'] = 'W/m^2'
meta_db['in/I_lw.latexUnits'] = r'W / {m}^2'
meta_db['in/I_lw.ymin'] = 0.0
meta_db['in/I_lw.ymax'] = 500.0

meta_db['in/T_a.Description'] = 'air temperature'
meta_db['in/T_a.Units'] = 'C'
meta_db['in/T_a.latexUnits'] = 'C'
meta_db['in/T_a.ymin'] = -18.1
meta_db['in/T_a.ymax'] = 32.9

meta_db['in/e_a.Description'] = 'vapor pressure'
meta_db['in/e_a.Units'] = 'Pa'
meta_db['in/e_a.latexUnits'] = 'Pa'
meta_db['in/e_a.ymin'] = 71.2
meta_db['in/e_a.ymax'] = 1606.4

meta_db['in/u.Description'] = 'wind speed'
meta_db['in/u.Units'] = 'm/sec'
meta_db['in/u.latexUnits'] = 'm/sec'
meta_db['in/u.ymin'] = 0.47
meta_db['in/u.ymax'] = 15.7

meta_db['in/T_g.Description'] = 'soil temperature at 0.5 m depth'
meta_db['in/T_g.Units'] = 'C'
meta_db['in/T_g.latexUnits'] = 'C'
meta_db['in/T_g.ymin'] = 0.0
meta_db['in/T_g.ymax'] = 1.3

meta_db['in/S_n.Description'] = 'net solar radiation'
meta_db['in/S_n.Units'] = 'W/m^2'
meta_db['in/S_n.latexUnits'] = r'W / {m}^2'
meta_db['in/S_n.ymin'] = 0.0
meta_db['in/S_n.ymax'] = 1200.0
         
# Precipitation
meta_db['ppt/m_pp.Description'] = 'total precipitation mass'
meta_db['ppt/m_pp.Units'] = 'kg/m^2'

meta_db['ppt/%_snow.Description'] = 'percentage of precipitation mass that was snow (0 to 1.0)'
meta_db['ppt/%_snow.Units'] = '%'

meta_db['ppt/rho_snow.Description'] = 'density of snowfall'
meta_db['ppt/rho_snow.Units'] = 'kg/m^3'

meta_db['ppt/T_pp.Description'] = 'average precipitation temperature'
meta_db['ppt/T_pp.Units'] = 'C'

# Energy & mass flux image
meta_db['em/R_n.Description'] = 'average net all-wave rad'
meta_db['em/R_n.Units'] = 'W/m^2'
meta_db['em/R_n.ymin'] = -111.5052185 
meta_db['em/R_n.ymax'] = 672.0316162

meta_db['em/H.Description'] = 'average sensible heat transfer'
meta_db['em/H.Units'] = 'W/m^2'
meta_db['em/H.ymin'] = -144.1464996
meta_db['em/H.ymax'] = 303.2650757

meta_db['em/L_v_E.Description'] = 'average latent heat exchange'
meta_db['em/L_v_E.Units'] = 'W/m^2'
meta_db['em/L_v_E.ymin'] = -319.4174194 
meta_db['em/L_v_E.ymax'] = 75.01908112

meta_db['em/G.Description'] = 'average snow/soil heat exchange'
meta_db['em/G.Units'] = 'W/m^2'
meta_db['em/G.ymin'] = 0.0, 
meta_db['em/G.ymax'] = 125.4

meta_db['em/M.Description'] = 'average advected heat from precipitation'
meta_db['em/M.Units'] = 'W/m^2'
meta_db['em/M.ymin'] = -14.62154675 
meta_db['em/M.ymax'] = 30.94813538

meta_db['em/delta_Q.Description'] = 'average sum of e.b. terms for snowcover'
meta_db['em/delta_Q.Units'] = 'W/m^2'
meta_db['em/delta_Q.ymin'] = -242.0096283 
meta_db['em/delta_Q.ymax'] = 677.6251221

meta_db['em/E_s.Description'] = 'total evaporation'
meta_db['em/E_s.Units'] = 'kg, or mm/m^2'
meta_db['em/E_s.ymin'] = -0.4 
meta_db['em/E_s.ymax'] = 1.0

meta_db['em/melt.Description'] = 'total melt'
meta_db['em/melt.Units'] = 'kg, or mm/m^2'
meta_db['em/melt.ymin'] = 0.0
meta_db['em/melt.ymax'] = 7.312501431

meta_db['em/ro_predict.Description'] = 'total melt'
meta_db['em/ro_predict.Units'] = 'kg, or mm/m^2'
meta_db['em/ro_predict.ymin'] = 0.0
meta_db['em/ro_predict.ymax'] = 10.30402946

meta_db['em/cc_s.Description'] = "snowcover cold content (energy required to bring snowpack's temperature to 273.16K)"
meta_db['em/cc_s.Units'] = 'J/m^2'
meta_db['em/cc_s.ymin'] = -5774940.0 
meta_db['em/cc_s.ymax'] = 1.0
