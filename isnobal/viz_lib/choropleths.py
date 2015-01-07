from os.path import join as joinpath
from copy import deepcopy
    
from isnobal.isnobalconst import IPW_IN, IPW_PPT, IPW_EM, IPW_SNOW


#
# IN
#

def netSolar(ipw_fn, dst_dir, mask=None, srcwin=None,
                     basename_template='%s.%04i.NetSolar.tif'):

    # Importing IPW here to avoid recursive import issues
    from isnobal import IPW

    # Importing just the needed map and norm funcs yields a 20%
    # performance increase
    from isnobal.viz_lib.colormaps import s_n_cm, s_n_norm

    ipw = IPW(ipw_fn)
    assert ipw.vtype in [IPW_IN]

    ipw['S_n'].data = s_n_norm(ipw['S_n'].data)
        
    basename = basename_template % (ipw.vtype, ipw.step)
    dst_fn = joinpath(dst_dir, basename)
    return ipw['S_n'].colorize(dst_fn, colormap=s_n_cm,
                        mask=mask, srcwin=srcwin, write_null=False)

def radiation2d(ipw_fn, dst_dir, mask=None, srcwin=None,
                     basename_template='%s.%04i.Radiation2d.tif'):
    import numpy as np
    from isnobal import IPW
    from isnobal.viz_lib.colormaps import radiation2d_cm

    ipw = IPW(ipw_fn)
    assert ipw.vtype in [IPW_IN]
    assert 'I_lw' in ipw

    rad = deepcopy(ipw['I_lw'])

    # for the 2-d color mapping the second dimension proportion
    # of Net Solar to total radiation the data array is cast as
    # a complex type and the values are packed in the imaginary
    # component. This avoids having to refactor Band.colorize as
    # we can just pass it a colormapping function that takes
    # complex values
    real = rad.data[:,:]
    real -= 150.0
    real *= 3.0

    imag = np.zeros(real.shape)
    
    if 'S_n' in ipw:
        imag  = ipw['S_n'].data
        imag /= 800.0

        indx = np.where(imag < 0.0)
        imag[indx] = 0.0
        
        indx = np.where(imag > 1.0)
        imag[indx] = 1.0
        
        real += ipw['S_n'].data

    real /= 800.0
    indx = np.where(real < 0.0)
    real[indx] = 0.0
    
    indx = np.where(real > 1.0)
    real[indx] = 1.0

    rad.data = np.zeros(real.shape, dtype=np.complex)
    rad.data.real = real
    rad.data.imag = imag

    basename = basename_template % (ipw.vtype, ipw.step)
    dst_fn = joinpath(dst_dir, basename)
    rad.colorize(dst_fn, colormap=radiation2d_cm,
                 mask=mask, srcwin=srcwin)

#
# SNOW
#

def snowWaterEquivalent(ipw_fn, dst_dir, mask=None, srcwin=None,
                                basename_template='%s.%04i.SWE.tif'):
    
    from isnobal import IPW
    from isnobal.viz_lib.colormaps import swe_cm, swe_norm_m

    ipw = IPW(ipw_fn)
    assert ipw.vtype in [IPW_SNOW]
    assert 'h2o_sat' in ipw

    swe = deepcopy(ipw['z_s'])
    swe.data *= ipw['h2o_sat'].data
    swe.data = swe_norm_m(swe.data)
        
    basename = basename_template % (ipw.vtype, ipw.step)
    dst_fn = joinpath(dst_dir, basename)
    swe.colorize(dst_fn, colormap=swe_cm, mask=mask,
                 maskedvalue=0.5,
                 srcwin=srcwin, write_null=False)
#
# PPT
#

def doppler(ipw_fn, dst_dir, mask=None, srcwin=None,
                    basename_template='%s.%04i.Doppler.tif'):
    
    from isnobal import IPW
    from isnobal.viz_lib.colormaps import doppler_mixed_cm, \
                                          doppler_snow_cm, \
                                          doppler_rain_cm, \
                                          doppler_norm_kg_per_sqm
                                              
    ipw = IPW(ipw_fn)
    assert ipw.vtype in [IPW_PPT]
    assert 'm_pp' in ipw
    
    m_pp = ipw['m_pp']
    m_pp.data = doppler_norm_kg_per_sqm(m_pp.data)
                
    # frames with mixed states never occur (with the data
    # that I have at least)
    p_snow = ipw['%_snow'].data[0,0]

    if p_snow < 1.0 and p_snow > 0.0:
        cm = doppler_mixed_cm
    elif p_snow >= 1.0:
        cm = doppler_snow_cm
    else:
        cm = doppler_rain_cm

    basename = basename_template % (ipw.vtype, ipw.step)
    dst_fn = joinpath(dst_dir, basename)
    m_pp.colorize(dst_fn, colormap=cm, mask=mask,
                  srcwin=srcwin, write_null=False)

#
# EM
#

def cummSurfaceWaterInputs(ipw_fns, dst_dir, ipw_ppt_fns,  
                   mask=None, srcwin=None,
                   basename_template='%s.%04i.CummSurfaceWaterInputs.tif'):

    from isnobal import IPW
    from isnobal.viz_lib.colormaps import swi_cm, swi_norm

    assert len(ipw_fns) == len(ipw_ppt_fns)

    ipw = IPW(ipw_fns[0])
    assert ipw.vtype in [IPW_EM]
    assert 'melt' in ipw
    assert 'ro_predict' in ipw

    swi = deepcopy(ipw['melt'])
    swi.data -= ipw['ro_predict'].data
    swi.data -= ipw['E_s'].data

    for fn in ipw_fns[1:]:
        ipw = IPW(fn)
        assert ipw.vtype in [IPW_EM]
        assert 'melt' in ipw
        assert 'ro_predict' in ipw

        swi.data += ipw['melt'].data
        swi.data -= ipw['ro_predict'].data
        swi.data -= ipw['E_s'].data

    for fn in ipw_ppt_fns:
        if fn is None:
            continue
        
        ipw = IPW(fn)
        assert ipw.vtype in [IPW_PPT]
        assert 'm_pp' in ipw

        swi.data += ipw['m_pp'].data

    swi.data = swi_norm(swi.data)
    
    ipw = IPW(ipw_fns[-1])        
    basename = basename_template % (ipw.vtype, ipw.step)
    dst_fn = joinpath(dst_dir, basename)
    swi.colorize(dst_fn, colormap=swi_cm,
                 mask=mask, srcwin=srcwin, write_null=False)
    
def surfaceWaterInputs(ipw_fn, dst_dir, ipw_ppt_fn=None, 
                mask=None, srcwin=None,
                basename_template='%s.%04i.SurfaceWaterInputs.tif'):

    assert dst_dir is not None
    
    cummSurfaceWaterInputs([ipw_fn], dst_dir, [ipw_ppt_fn],  
                           mask=mask, srcwin=srcwin,
                           basename_template=basename_template)



def melt(ipw_fn, dst_dir, mask=None, srcwin=None,
                     basename_template='%s.%04i.Melt.tif'):

    from isnobal import IPW
    from isnobal.viz_lib.colormaps import melt_cm, melt_norm

    ipw = IPW(ipw_fn)
    assert ipw.vtype in [IPW_EM]
    assert 'melt' in ipw

    ipw['melt'].data = melt_norm(ipw['melt'].data)
        
    basename = basename_template % (ipw.vtype, ipw.step)
    dst_fn = joinpath(dst_dir, basename)
    ipw['melt'].colorize(dst_fn, colormap=melt_cm,
                        mask=mask, srcwin=srcwin, write_null=False)

def delta_Q(ipw_fn, dst_dir, mask=None, srcwin=None,
                     basename_template='%s.%04i.delta_Q.tif'):

    from isnobal import IPW
    from isnobal.viz_lib.colormaps import delta_Q_cm, delta_Q_norm

    ipw = IPW(ipw_fn)
    assert ipw.vtype in [IPW_EM]
    assert 'delta_Q' in ipw
    
    ipw['delta_Q'].data = delta_Q_norm(ipw['delta_Q'].data)
        
    basename = basename_template % (ipw.vtype, ipw.step)
    dst_fn = joinpath(dst_dir, basename)
    ipw['delta_Q'].colorize(dst_fn, colormap=delta_Q_cm,
                            mask=mask, srcwin=srcwin, write_null=False)
