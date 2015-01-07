from __future__ import print_function

# Copyright (c) 2014, Roger Lew (rogerlew.gmail.com)
#
# The project described was supported by NSF award number IIA-1301792
# from the NSF Idaho EPSCoR Program and by the National Science Foundation.



"""
vtype in [IPW_IN, IPW_PPT, IPW_EM, IPW_SNOW] == True
vname refers to band names, e.g. "S_n"

ISNOBAL (object)
 - fns (dict)
 |   - [vtype] -> <list of IPW input files>
 |
 - steps (dict)
 |   - [vtype] -> <ndarray of ascending int32 sim steps> 
 |
 - pvs (dict)
 |   - [vtype] (dict)
 |       + [vname] (OrderedDict)
               -> ProcessVariable (object)
 |                  + parent (*ISNOBAL)
 |                        '''Access to ISNOBAL metadata and methods'''
 |                  + <hd5f spatio-temporal interface>
 |                ------------------------------------------------------
 |                  + read2dData() -> 2d ndarray (nx*ny, nt)
 |                  + read3dData() -> 3d ndarray (nx, ny, nt)
 |                  + aspectEnsemble()
 |                       '''Produces ensemble timeseries'''
 |
------------------------------------------------------------------------
 + packToHd5()
 |     '''packs IPW data to hdf5 container to support pvs interface'''
 
 
IPW (object)
 + <metadata>
 - bands (dict)
 |   - [vname] -> Band (object)
 |                  + parent (*IPW)
 |                        '''Access to IPW metadata and methods'''
 |                  + <metadata>
 |                  - data (ndarray)
 |                ------------------------------------------------------
 |                  + __init__()
 |                        '''Reads IPW and GTIFF'''
 |                  + buildMappingFromData()
 |                        '''identifies scale mapping for writing'''
 |                  + colorize()
 |                        '''used to produce RGBA viz output'''
 |
------------------------------------------------------------------------
 + write()
 |     '''Writes IPW'''
 + translate()
 |     '''Writes GTiff'''
 |

(http://zero.eng.ucmerced.edu/snow/whitney/ly/www/man1/isnobal.html)
Constants, assumed over the grid:

     max_z_s_0   = 0.25 m   Thickness of the active layer
     z_u         = 5.0 m    Height above the ground of the wind
                            speed measurement
     z_T         = 5.0 m    Height above the ground of the air
                            temperature and vapor pressure
                            measurements


Initial conditions image (7-band):

     z         =   elevation (m)
     z_0       =   roughness length (m)
     z_s       =   total snowcover depth (m)
     rho       =   average snowcover density (kg/m^3)
     T_s_0     =   active snow layer temperature (C)
     T_s       =   average snowcover temperature (C)
     h2o_sat   =   % of liquid H2O saturation (relative
                   water content, i.e., ratio of water in
                   snowcover to water that snowcover could
                   hold at saturation)

                   
If the -r option (restart), the image has an additional band for the lower snow
layer's temperature, T_s_l, (8-bands):

     z         =   elevation (m)
     z_0       =   roughness length (m)
     z_s       =   total snowcover depth (m)
     rho       =   average snowcover density (kg/m^3)
     T_s_0     =   active snow layer temperature (C)
     T_s_l     =   temperature of the snowpack's lower
                   layer (C)
     T_s       =   average snowcover temperature (C)
     h2o_sat   =   % of liquid H2O saturation (relative
                   water content, i.e., ratio of water in
                   snowcover to water that snowcover could
                   hold at saturation)


Precipitation image (4-band):

     m_pp       =   total precipitation mass (kg/m^2)
     %_snow     =   % of precipitation mass that was snow (0 to 1.0)
     rho_snow   =   density of snowfall (kg/m^3)
     T_pp       =   average precip. temperature (C) (from dew point
                    temperature if available, or can be estimated
                    from air temperature during storm, or minimum
                    daily temperature)

Like the point model, the DEM-based model will parse mixed rain/snow events.
It is designed to accept inputs that could be derived from typical NRCS Snotel
data such as total precipitation, snow mass increase, and temperature. The user
must estimate average density and percent snow if depth data are unavailable.
The model makes the following assumptions about the snow temperature, rain
temperature, and liquid water saturation of the snow:

     when 0.0 < %_snow < 1.0, (a mixed rain/snow event)
          snow temperature = 0.0
          rain temperature = T_pp
          liquid H2O sat.  = 100%
     when %_snow = 1.0 and T_pp => 0.0, (a warm snow-only event)
          snow temperature = 0.0
          liquid H2O sat.  = 100%
     when %_snow = 1.0 and T_pp < 0.0, (a cold snow event)
          snow temperature = T_pp
          liquid H2O sat.  = 0%


Input image (6-band):

     I_lw   =   incoming thermal (long-wave) radiation (W/m^2)
     T_a    =   air temperature (C)
     e_a    =   vapor pressure (Pa)
     u      =   wind speed (m/sec)
     T_g    =   soil temperature at 0.5 m depth (C)
     S_n    =   net solar radiation (W/m^2)


Energy & mass flux image (10-band):

     R_n          =   average net all-wave rad (W/m^2)
     H            =   average sensible heat transfer (W/m^2)
     L_v_E        =   average latent heat exchange (W/m^2)
     G            =   average snow/soil heat exchange (W/m^2)
     M            =   average advected heat from precip. (W/m^2)
     delta_Q      =   average sum of e.b. terms for snowcover (W/m^2)
     E_s          =   total evaporation (kg, or mm/m^2)
     melt         =   total melt (kg, or mm/m^2)
     ro_predict   =   total predicted runoff (kg, or mm/m^2)
     cc_s         =   snowcover cold content (energy required to
                      bring snowpack's temperature to 273.16K)
                      (J/m^2)
                      

Snow conditions image (9-band):

     z_s       =   predicted depth of snowcover (m)
     rho       =   predicted average snow density (kg/m^3)
     m_s       =   predicted specific mass of snowcover (kg/m^2)
     h2o       =   predicted liquid H2O in snowcover (kg/m^2)
     T_s_0     =   predicted temperature of surface layer (C)
     T_s_l     =   predicted temperature of lower layer (C)
     T_s       =   predicted average temp of snowcover (C)
     z_s_l     =   predicted lower layer depth (m)
     h2o_sat   =   predicted % liquid h2o saturation
"""

__version__ = "0.0.2"

from copy import deepcopy
from collections import OrderedDict
from datetime import datetime, timedelta
from glob import glob
from itertools import izip
import os
import shutil
from os.path import join as joinpath
from pprint import pprint as pp
from cStringIO import StringIO
import math
import random
import time
import warnings

import h5py
import numpy as np
from numpy.testing import assert_array_almost_equal

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap

import shapefile

from osgeo import gdal, ogr, osr
from osgeo.gdalconst import *

from isnobal.isnobalconst import *
from isnobal.lib import clean, identifyStep


class VarlistManager:
    """
    singleton class that manages the groups and variables
    of the ISNOBAL model
    """
    
    def __init__(self):
        _d ={}
        
        _d[IPW_DEM] = ('dem',)
        _d[IPW_MASK] = ('mask',)
        _d[IPW_INIT] = ('z', 'z_0', 'z_s', 'rho',
                        'T_s_0', 'T_s_l', 'T_s', 'h2o_sat')
        _d[IPW_IN] = ('I_lw', 'T_a', 'e_a', 'u', 'T_g', 'S_n')
        _d[IPW_PPT] = ('m_pp', '%_snow', 'rho_snow', 'T_pp')
        _d[IPW_EM] = ('R_n', 'H', 'L_v_E', 'G', 'M', 'delta_Q',
                      'E_s', 'melt', 'ro_predict', 'cc_s')
        _d[IPW_SNOW] = ('z_s', 'rho', 'm_s', 'h2o', 'T_s_0',
                        'T_s_l', 'T_s', 'z_s_l', 'h2o_sat')

        self._d = _d
        
    def __call__(self, fname, nbands):
        """
        determines band variable type from ffname and
        returns cooresponding variable names
        """
        _d = self._d
        
        basename = os.path.basename(fname)

        if IPW_DEM in basename.lower():
            vtype = IPW_DEM
        elif IPW_MASK in basename.lower():
            vtype = IPW_MASK
        elif IPW_INIT in basename.lower():
            vtype = IPW_INIT
        elif IPW_IN in basename.lower():
            vtype = IPW_IN
        elif IPW_PPT in basename.lower():
            vtype = IPW_PPT
        elif IPW_EM in basename.lower():
            vtype = IPW_EM
        elif IPW_SNOW in basename.lower():
            vtype = IPW_SNOW
        else:
            vtype = None

        if vtype is None:
            varlist = ['band%02i'%i for i in xrange(nbands)]
        else:
            varlist = _d[vtype]
        
        assert len(varlist) >= nbands, \
               print((fname, len(varlist), nbands, varlist))

        return vtype, varlist

    def getBandType(self, band):
        """
        Given a band determines and returns the band type
        """
        _d = self._d
        
        if band in _d[IPW_IN]:
            return IPW_IN
        elif band in _d[IPW_PPT]:
            return IPW_PPT
        elif band in _d[IPW_EM]:
            return IPW_EM
        elif band in _d[IPW_SNOW]:
            return IPW_SNOW

        return None

    def __getitem__(self, band):
        """
        Given a band determines and returns tuple of variable names
        """
        return self._d[band]
    
    def __setitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        """
        iterator yielding group and band names for input,
        precipitation, energy mass flux, and snow variables
        """
        for grp in [IPW_IN, IPW_PPT, IPW_EM, IPW_SNOW]:
            for band in self[grp]:
                yield grp, band
    
varlistManager = VarlistManager()

#
# Band
#

_unpackindx  = lambda L: int(L.split()[2])
_unpackint   = lambda L: int(L.split('=')[1].strip())
_unpackfloat = lambda L: float(L.split('=')[1].strip())
_unpackstr   = lambda L: L.split('=')[1].strip()


class Band:
    """
    Represents a raster band of geospatial data
    """
    def __init__(self, nlines, nsamps, parent, pseudo=False):
        
        self.parent = parent
        self.pseudo = pseudo # used as flag to control which
                             # bands are written to file
        
        # width, height
        self.nlines, self.nsamps = nlines, nsamps

        # basic_image
        self.name = None
        self.bytes = None
        self.fmt = None
        self.bits = None
        self.annot = None
        self.history = []

        # geo
        self.bline = None
        self.bsamp = None
        self.dline = None
        self.dsamp = None
        self.geounits = None
        self.coord_sys_ID = None
        self.geotransform = None

        # lq
        self.x0 = None
        self.xend = None
        self.y0 = None
        self.yend = None
        self.units = None
        self.transform = None

        self.data = None

    def _parse_geo(self, L0, L1, L2, L3, L4, L5):
        """
        sets attributes and builds GDAL ordered
        geotransform list
        """
        bline = self.bline = _unpackfloat(L0)
        bsamp = self.bsamp = _unpackfloat(L1)
        dline = self.dline = _unpackfloat(L2)
        dsamp = self.dsamp = _unpackfloat(L3)
        self.geounits = _unpackstr(L4)
        self.coord_sys_ID = _unpackstr(L5)
        self.geotransform = [bsamp - dsamp / 2.0,
                             dsamp, 0.0,
                             bline - dline / 2.0,
                             0.0, dline]

    def _set_mapping(self, x0, y0, xend, yend):
        self.transform = lambda x: (yend - y0) * (x / xend) + y0
        self.untransform = \
                lambda: np.array(
                            np.round(xend * ((self.data - y0) / (yend-y0))),
                            dtype=self.fmt)
        self.x0, self.xend = x0, xend
        self.y0, self.yend = y0, yend
        
    def _parse_lq(self, L0, L1):
        """
        Pulls the scale values from line0 and line1 builds
        a function for transforming integer values to floats
        """
        [[x0, y0], [xend, yend]] = [L0.split(), L1.split()]
        self._set_mapping(float(x0), float(y0), float(xend), float(yend))

    def buildMappingFromData(self):
        """
        Determines scale mapping based off of minimum and maximum
        values of the data for writing to IPW as uint types.
        """
        self._set_mapping(self.x0, self.minimum(), self.xend, self.maximum())

    def calculateSurfaceNormals(self, scale=1):
        """
        Builds and returns a 3d ndarray of elevation surface.
        Band should represent elevation data

        Parameters
        ----------
        scale : float (default=1)
            multiplicative scale applied to elevation data

        Returns
        -------
        normal_map : ndarray 
            shape is (nsamps, nlines, 3)
            3rd dimension is ordered: dx, dy, dz
            
        """

        if self.name not in ['z', 'dem']:
            warnings.warn('Are you sure band data represents elevation?\n'
                          'Band name is: "%s"' % self.name)
        
        xres, yres = self.dsamp, self.dline
        elevation = self.data * scale
        
        # gradient in x and y directions
        dy, dx = np.gradient(elevation)  # find gradient
        dx /=  -xres
        dy /=  -yres
        
        slope = np.arctan(np.hypot(dx, dy))
        dz = np.cos(slope)
        
        d = np.sqrt(dx**2 + dy**2 + dz**2)
        
        return dx/d, dy/d, dz/d
    
    def minimum(self):
        """
        returns the minimum of the band data
        """
        return np.min(self.data)

    def maximum(self):
        """
        returns the maximum of the band data
        """
        return np.max(self.data)

    def colorize(self, dst_fname, colormap, drivername='Gtiff',
                 mask=None, srcwin=None, write_null=True):
        """
        colorize(band, colormap[, ymin=None][, ymax=None])

        Build a colorized georeferenced raster of a band.

        Assumes the band data has been normalized between [0-1]

        Parameters
        ----------
        colormap : string or colormap object
            string name of a matplotlib.colors colormap
            colormap object should take a float between 0 - 1 and
            return an RGBA tuple with channel values between 0 - 1.

        drivername : string (default='Gtiff')
            GDAL drivername for output

        mask : None or ndarray
            if ndarray should be same shape as self.data

        srcwin : None or list of ints
            list specifies [xoff, yoff, xsize, ysize]
            (same order as gdal_translate)

        write_null : bool
            false -> no dataset is written if the alpha channel is all 0.0
            true -> dataset is always written

        Returns
        -------
        dst_fname : string or None
        """

        _dir, basename = os.path.split(dst_fname)

        # make sure the path exists
        if _dir != '' and not os.path.exists(_dir):
            print(os.path.exists(_dir))
            raise Exception('"%s" directory does not exist' % _dir)
        
        nsamps, nlines = int(self.nlines), int(self.nsamps)
        parent = self.parent
        epsg = parent.epsg

        # find colormap
        if isinstance(colormap, basestring):
            cm = plt.get_cmap(colormap)
        else:
            cm = colormap

        data = self.data#[:,:]

        assert np.max(data) <= 1.0, print('Data max > 1.0: %f' % np.max(data))
        assert np.min(data) >= 0.0, print('Data min < 0.0: %f' % np.min(data))

        # cast data as masked array if mask is supplied
        if mask is not None:
            assert mask.shape == data.shape
            data = np.ma.MaskedArray(data, 1.0-mask)

        # crop data if sourcewin is supplied
        if srcwin is not None:
            xoff, yoff, xsize, ysize = srcwin
            data = data[yoff:yoff+ysize,
                        xoff:xoff+xsize]

            if mask is not None:
                mask = mask[yoff:yoff+ysize,
                            xoff:xoff+xsize]

            # change nsamps and nlines for geotransform
            nsamps = xsize
            nlines = ysize

        
        # use colormap to find rgba byte values
        rgba = np.array(cm(data) * 255.0, dtype=np.uint8)

        if mask is not None:
            rgba[:, :, 3] *= mask
            
        if not write_null:
            if np.sum(rgba[:, :, 3]) == 0:
                return None
        
        # find geotransform
        for b in parent:
            gt0 = b.geotransform
            if gt0 is not None:
                break

        # initialize raster
        driver = gdal.GetDriverByName(drivername)
        ds = driver.Create(dst_fname, nlines, nsamps, 
                           4, GDT_Byte)

        if ds is None:
            raise Exception('Could not open "%s"' % dst_fname)
        
        # set projection
        if epsg is not None:
            proj = osr.SpatialReference()
            status = proj.ImportFromEPSG(parent.epsg)
            if status != 0:
                warnings.warn('Importing epsg %i return error code %i'
                              % (epsg, status))
            ds.SetProjection(proj.ExportToWkt())

        # set geotransform
        if gt0 is None:
            warnings.warn('Unable to find a geotransform')
        else:
            
            if srcwin is not None:
                gt0[0] += xoff * gt0[1]
                gt0[3] += yoff * gt0[5]
                
            # set transform
            ds.SetGeoTransform(gt0)

        # write data
        for i in xrange(4):
            ds.GetRasterBand(i+1).WriteArray(rgba[:, :, i])

        ds = None  # Writes and closes file

        return dst_fname

    def __str__(self):
        return  '  nlines (width): {0.nlines}\n' \
                '  nsamps (height): {0.nsamps}\n' \
                '  bytes: {0.bytes}\n' \
                '  transform: {0.x0} -> {0.xend}\n' \
                '             {0.y0} -> {0.yend}\n' \
                '  geo: {0.bline} {0.bsamp} {0.dline} {0.dsamp}\n' \
                .format(self)


#
# IPW
#

# templates for writing IPW
_h0_template = """\
!<header> basic_image_i -1 $Revision: 1.11 $
byteorder = {0.byteorder} 
nlines = {0.nlines} 
nsamps = {0.nsamps} 
nbands = {0.nbands}
"""

_h_basic_image_template = """\
!<header> basic_image {0} $Revision: 1.11 $
bytes = {1.bytes} 
bits = {1.bits}
"""

_h_lq_template = """\
!<header> lq {0} $Revision: 1.6 ${2}
map = {1.x0} {1.y0} 
map = {1.xend} {1.yend}
"""

_h_geo_template = """\
!<header> geo {0} $Revision: 1.7 $
bline = {1.bline} 
bsamp = {1.bsamp} 
dline = {1.dline} 
dsamp = {1.dsamp} 
units = {1.geounits} 
coord_sys_ID = {1.coord_sys_ID}
"""

##def _find_fmt(band):
##    if band.bits not in [8, 16]:
##        return 'uint16'
##        return '>u2'
##        return '>i%i' % band.bytes
##    else:
##        return ('uint8', 'uint16')[band.bytes == 2]
    
class IPW:
    """
    Represents a IPW file container
    """
    def __init__(self, fname, rescale=True, epsg=26911, varnames=None):
        """
        IPW(fname[, rescale=True])

        Parameters
        ----------
        fname : string
           path to the IPW container
           if "in", "ppt", "em", or "snow" are in fname
           band names will be pulled from VARNAME_DICT
           
        rescale : bool (default = True)
            Specifies whether data should be rescaled to singles
            based on the lq map or whether they should remain uint8/16

        epsg : int (default = 26911; UTM11N: NAD83)


        """
        if fname.lower().endswith('.tif'):
            self._parseTIF(fname, epsg, varnames)
        else:
            self._parseIPW(fname, rescale, epsg, varnames)

        # identify step number
        if self.vtype in [IPW_DEM, IPW_MASK]:
            self.step = 0
        else:
            self.step = identifyStep(fname)

        if self.vtype == IPW_IN and 'S_n' not in self:
            self._generate_sundown_S_n_Band()

    def _generate_sundown_S_n_Band(self):
        nlines, nsamps = self.nlines, self.nsamps
        
        b = Band(nlines, nsamps, parent=self, pseudo=True)
        b.bytes = 2
        b.fmt = 'uint16'
        b.bits = 11
        b._set_mapping(0, 0, 2047, 1200)
        b.data = np.zeros((nlines, nsamps))
        self.registerBand(b, 'S_n')
        
    def registerBand(self, band, name):
        band.name = name
        self.bands.append(band)
        self.name_dict[name] = self.nbands
        self.nbands += 1
        
    def _parseHeader(self, fid, st_size):
            
        readline = fid.readline  # dots make things slow (and ugly)
        tell = fid.tell

        bands = []
        byteorder = None
        nlines = None
        nsamps = None
        nbands = None
            
        while 1:  # while 1 is faster than while True
            line = readline()

            # fail safe, haven't needed it, but if this is running
            # on a server not a bad idea to have it here
            if tell() == st_size:
                return bands, byteorder, nlines, nsamps, nbands
            
            if '!<header> basic_image_i' in line:
                byteorder = _unpackstr(readline())
                nlines = _unpackint(readline())
                nsamps = _unpackint(readline())
                nbands = _unpackint(readline())

                # initialize the band instances in the bands list
                bands = [Band(nlines, nsamps, self) for j in xrange(nbands)]

            elif '!<header> basic_image' in line:
                indx = _unpackindx(line)
                bands[indx].bytes = _unpackint(readline())
                bands[indx].bits = _unpackint(readline())
                bands[indx].fmt = ('uint8', 'uint16')[bands[indx].bytes == 2]
                
                while line[0] != '!':
                    bands[indx].history.append(_unpackstr(line))

            elif '!<header> geo' in line:
                indx = _unpackindx(line)
                bands[indx]._parse_geo(*[readline() for i in xrange(6)])

            elif '!<header> lq' in line:
                indx = _unpackindx(line)
                line1 = fid.readline()
                if 'units' in line1:
                    bands[indx].units = _unpackstr(line1)

                    bands[indx]._parse_lq(_unpackstr(readline()),
                                          _unpackstr(readline()))
                else:
                    bands[indx]._parse_lq(_unpackstr(line1),
                                          _unpackstr(readline()))

            if '\f' in line:  # feed form character separates the
                break         # image header from the binary data


        return bands, byteorder, nlines, nsamps, nbands

    def _parseTIF(self, fname, epsg, varnames):
        ds = gdal.Open(fname)

        if ds is None:
            raise Exception('Could not open "%s"' % fname)

        header = ds.GetMetadataItem('IPW Header')
        if header is None:
            raise Exception('"%s" does not contain IPW Header data' % fname)
        fid = StringIO(header)
        bands, byteorder, nlines, nsamps, nbands = \
               self._parseHeader(fid, len(header))
        fid.close()
        
        for i in xrange(nbands):
            b = ds.GetRasterBand(i+1)
            bands[i].data = b.ReadAsArray()

        rescale = b.DataType == GDT_Float32

        ds = None  # Writes and closes file

        # attempt to assign names to the bands
        assert nbands == len(bands)

        if varnames is None:
            vtype, varlist = varlistManager(fname, nbands)
        else:
            vtype = None
            varlist = varnames
            
        for b, name in zip(bands, varlist[:nbands]):
            b.name = name

        assert nbands <= len(varlist)
        
        self.vtype = vtype
        self.epsg = epsg
        self.header = header
        self.fname = fname
        self.rescale = rescale
        self.name_dict = dict(zip(varlist, range(nbands)))
        self.bip = None
        self.bands = bands
        self.byteorder = byteorder
        self.nlines = nlines
        self.nsamps = nsamps
        self.nbands = nbands
        
    def _parseIPW(self, fname, rescale, epsg, varnames):
    
        # open data file
        fid = open(fname, 'rb')

        # read header
        st_size = os.fstat(fid.fileno()).st_size
        bands, byteorder, nlines, nsamps, nbands = \
               self._parseHeader(fid, st_size)

        # store where the header ends
        ff_byte = fid.tell()

        # attempt to assign names to the bands
        assert nbands == len(bands)

        if varnames is None:
            vtype, varlist = varlistManager(fname, nbands)
        else:
            vtype = None
            varlist = varnames

        if vtype == IPW_INIT:
            if nbands == 7:
                varlist = tuple(v for v in varlist if v != 'T_s_l')

        assert nbands <= len(varlist)

        for b, name in zip(bands, varlist[:nbands]):
            b.name = name

        # Unpack the binary data using numpy.fromfile
        # because we have been reading line by line fid is at the
        # first data byte, we will read it all as one big chunk
        # and then put it into the appropriate bands
        #
        # np.types allow you to define heterogenous arrays of mixed
        # types and reference them with keys, this helps us out here
        
        dt = np.dtype([(b.name, b.fmt) for b in bands])

        bip = sum([b.bytes for b in bands])  # bytes-in-pixel
        required_bytes = bip * nlines * nsamps
        assert st_size >= required_bytes

        # this is way faster than looping with struct.unpack
        # struct.unpack also starts assuming there are pad bytes
        # when format strings with different types are supplied
        data = np.fromfile(fid, dt, count=nlines*nsamps)

        # Separate into bands
        data = data.reshape(nlines, nsamps)
        for b in bands:
            if rescale:
                try:
                    b.data = np.array(b.transform(data[b.name]),
                                      dtype=np.float32)
                except:
                    for v in data[b.name][:10,0]:
                        print( v)
                    
            else:
                b.data = np.array(data[b.name],
                                  dtype=np.dtype(b.fmt))

        fid.seek(0)
        self.header = fid.read(ff_byte)
        self.vtype = vtype
        self.epsg = epsg   
        self.fname = fname
        self.rescale = rescale
        self.name_dict = dict(zip(varlist, range(nbands)))
        self.bip = bip
        self.bands = bands
        self.byteorder = byteorder
        self.nlines = nlines
        self.nsamps = nsamps
        self.nbands = nbands

        fid.close()

    def __getitem__(self, key):
        try:
            return self.bands[key]
        except:
            return self.bands[self.name_dict[key]]

    def __iter__(self):
        for b in self.bands:
            yield b

    def __contains__(self, key):
        return key in self.name_dict
        
    def write(self, dst_name, warn_on_overwrite=False,
              write_pseudo_bands=False):
        global _h0_template
        global _h_basic_image_template
        global _h_lq_template
        global _h_geo_template

        rescale = self.rescale
        nlines = self.nlines
        nsamps = self.nsamps

        # handle write_pseudo_bands option
        write_bands = []
        for b in self:
            if ((not write_pseudo_bands) and (b.pseudo)):
                continue
            write_bands.append(b)
            
        # open dst_name
        if os.path.exists(dst_name) and warn_on_overwrite:
            warnings.warn('"dst_name" exists and will be overwritten')
        fid = open(dst_name, 'wb')

        # write header
        fid.write(_h0_template.format(self))

        for i, b in enumerate(write_bands):
            fid.write(_h_basic_image_template.format(i, b))

        for i, b in enumerate(write_bands):
            if rescale:
                ymin = b.minimum()
                ymax = b.maximum()
                if round(ymin, 1) < round(b.y0, 1) or \
                   round(ymax, 1) > round(b.yend, 1):
                    b._set_mapping(0, ymin, 2**b.bits-1, ymax)
                
            _units = ''
            if b.units is not None:
                _units = '\nunits = %s ' % b.units
            fid.write(_h_lq_template.format(i, b, _units).replace('.0 ', ' '))
            
        for i, b in enumerate(write_bands):
            if any((b.bline is None, b.bsamp is None, b.dline is None,
                    b.geounits is None, b.coord_sys_ID is None)):
                continue

            fid.write(_h_geo_template.format(i, b))
            
        fid.write('!<header> image -1 $Revision: 1.5 $\f\n')

        # format and write data
        dt = np.dtype([(b.name, b.fmt) for b in write_bands])
        data = np.zeros((nlines, nsamps), dtype=dt)

        for b in write_bands:
            if rescale:
                data[b.name] = b.untransform()
            else:
                data[b.name] = b.data

        fid.write(data.tostring('C'))

        # close file
        fid.close()
                    
    def translate(self, dst_fname, writebands=None,
                  drivername='Gtiff', multi=False):
        """
        translate(dst_dataset[, bands=None][, drivername='GTiff']
                  [, multi=True])

        translates the data to a georeferenced tif.

        If the data has been rescaled all bands are written as
        Float 32. If the data has not been scaled the type is
        Uint8 if all channels are Uint8 and Uint16 otherwise

        Parameters
        ----------
        dst_fname : None or string
           path to destination file without extension.
           Assumes folder exists.

        writebands : None or iterable of integers
            Specifies which bands to write to file.
            Bands are written in the order specifed.

            If none, all bands will be written to file
            The first band is "0" (like iSNOBAL, not like GDAL)

        multi : bool (default True)
            True write each band to its own dataset
            False writes all the bands to a single dataset
        """

        _dir, basename = os.path.split(dst_fname)
        if _dir != '':
            assert os.path.exists(_dir)
            
        if writebands is None:
            writebands = range(self.nbands)

        if multi:
            for i in writebands:
                self._translate(dst_fname + '.%02i'%i, [i], drivername)
        else:
            self._translate(dst_fname, writebands, drivername)

    def _translate(self, dst_fname, writebands=None, drivername='Gtiff'):
        epsg = self.epsg
        rescale = self.rescale
        header = self.header
        bands = self.bands
        nbands = self.nbands
        nlines, nsamps = self.nlines, self.nsamps

        if writebands is None:
            writebands = range(nbands)

        num_wb = len(writebands)

        assert num_wb >= 1

        # The first band of the inputs doesn't have a
        # geotransform. I'm sure this is a feature and not a bug ;)
        #
        # Check to make sure the defined geotransforms are the same
        #
        # Haven't really found any cases where multiple bands have
        # different projections. Is this really a concern?
        gt_override = 0

        # search write bands for valid transform
        for i in writebands:
            gt0 = bands[i].geotransform
            if gt0 is not None:
                break

        if gt0 is None:
            # search all bands for valid transform
            for b in bands:
                gt0 = b.geotransform
                if gt0 is not None:
                    gt_override = 1
                    break
            if gt0 is None:
                raise Exception('No Projection Found')
            else:
                warnings.warn('Using Projection from another band')

        if not gt_override:
            for i in writebands:
                gt = bands[i].geotransform
                if gt is None:
                    continue
                assert_array_almost_equal(gt0, gt)

        # If the data has been rescaled all bands are written as
        # Float 32. If the data has not been scaled the type is
        # Uint8 if all channels are Uint8 and Uint16 otherwise
        if rescale:
            gdal_type = GDT_Float32
        else:
            if all([bands[i].bytes == 1 for i in writebands]):
                gdal_type = GDT_Byte
            else:
                gdal_type = GDT_UInt16

        # initialize raster
        driver = gdal.GetDriverByName(drivername)
        ds = driver.Create(dst_fname + '.tif', nsamps, nlines,
                           num_wb, gdal_type)

        if ds is None:
            raise Exception('Failed to create "%s"' % dst_fname)

        # set projection
        if epsg is not None:
            proj = osr.SpatialReference()
            status = proj.ImportFromEPSG(epsg)
            if status != 0:
                warnings.warn('Importing epsg %i return error code %i'
                              %(epsg, status))
            ds.SetProjection(proj.ExportToWkt())

        # set geotransform
        ds.SetGeoTransform(gt0)

        # write data
        for i in writebands:
            ds.GetRasterBand(i+1).WriteArray(bands[i].data)

        # write header
        ds.SetMetadataItem('IPW Header', header)

        ds = None  # Writes and closes file
        
    def __str__(self):
        s = ['IPW({0.fname})\n'
             '{1}\n'
             'byteorder: {0.byteorder}\n'
             'nlines: {0.nlines}\n'
             'nsamps: {0.nsamps}\n'
             'nbands: {0.nbands}\n'
             .format(self, '-'*57)]

        for i, b in enumerate(self.bands):
            s.append('\nBand %i\n'%i)
            s.append(str(b))

        return ''.join(s)

#
# ProcessVariable
#

def findDateTimeIndxs(datetimes, t0, tend):
    """
    determines start and stop indices lying within
    ascending datetimes list

    Parameters
    ----------
    datetimes : list of datetime.datetime objects
        should be sorted in ascending order

    t0 : datetime.datetime
    
    tend : datetime.datetime

    Returns
    -------
    j0 : int
         smallest index of datetimes that lies between t0 and tend
    jend : int
         largest index of datetimes that lies between t0 and tend
    """
    
    n = len(datetimes)
    
    j0 = None
    jend = None

    for i in xrange(n):
        
        if (t0 <= datetimes[i]) and (j0 is None):
            j0 = i

        if (tend >= datetimes[n - (i + 1)]) and (jend is None):
            jend = n - (i + 1)

        if (j0 is not None) and (jend is not None):
            return j0, jend
            
class ProcessVariable:
    """
    simplistic interface for reading process variables
    from an HDF5 container.

    must first use the ISNOBAL.packToHd5 method to
    use this interface.
    """
    def __init__(self, band, parent):
        
        vtype = self.vtype = varlistManager.getBandType(band)
        self.band = band
        self.parent = parent
        self.shape = (parent.ysize,
                      parent.xsize,
                      len(parent.fns[vtype]))
        
        startdate = parent.startdate
        stepdelta = parent.stepdelta
        self.datetimes = []
        for step in parent.steps[vtype]:
            self.datetimes.append(startdate + step * stepdelta)

        self.min = None
        self.max = None
        self.avg = None
        self.std = None
        
    def __iter__(self):
        data3d = self.read3dData()
        nx,ny,nt = data3d.shape

        for i in xrange(nt):
            yield data3d[:,:,i]
            
    def calculateSummaryStats(self):
        data = self.read2dData()

        self.min = np.min(data)
        self.max = np.max(data)
        self.avg = np.mean(data)
        self.std = np.std(data)

        shape = (self.parent.ysize, self.parent.xsize)
        self.min_spatial = np.min(data, axis=1).reshape(shape)
        self.max_spatial = np.max(data, axis=1).reshape(shape)
        self.avg_spatial = np.mean(data, axis=1).reshape(shape)
        self.std_spatial = np.std(data, axis=1).reshape(shape)

        self.min_temporal = np.min(data, axis=0)
        self.max_temporal = np.max(data, axis=0)
        self.avg_temporal = np.mean(data, axis=0)
        self.std_temporal = np.std(data, axis=0)
        
    def read2dData(self):
        if self.parent.hasHd5():
            root = h5py.File(self.parent.hdf5_fname)
            path = self.vtype + r'/' + self.band
            data = root[path][:]
            root = None

            return data
        else:
            raise NotImplementedError('must pack ISNOBAL to hd5')

    def read3dData(self):
        if self.parent.hasHd5():
            return self.read2dData().reshape(self.shape)
        else:
            raise NotImplementedError('must pack ISNOBAL to hd5')

    def aspectEnsemblePlot(self, dst_name, figsize=None, dpi=100,
                           numsteps=None, steps_per_row=168,
                           line_alpha=0.04, subplots_adjust_kwargs=None,
                           ylim=None, cell_sample_percentage=0.05,
                           verbose=True, scatter=False, apply_mask=False):
        """
        Builds an ensemble plot of process variable data.

        The seperate lines of the ensemble are sampled from the non-masked
        cells and represent the topological aspect of the cell.

        Parameters
        ----------
        dst_fname : string
            name of output file

        figsize : None or tuple (width, height)
            if None the width is set at 20" and the height is determined
            by the number of rows

        dpi : 100 resolution of the output

        subplots_adjust_kwargs: None or dict
            if None the top and bottom are set to have a 1/2" margin

        ylim : None or list of floats [ymin, ymax]
            applied across all rows (subplots)

        cell_sample_percentage : float (0-1]
            controls percentage of non-masked cells that are randomly
            sampled to be represented in ensemble

        scatter : boolean
            specifies whether ensemble is plotted as line or scatter

        apply_mask : boolean
            specifies whether the parent's mask should be applied to the data

        Returns
        -------
        None
        """

        # unpack variables
        mask = self.parent.mask
        vtype = self.vtype
        nx, ny, nt = self.shape
        datetimes = self.datetimes
        startdate = self.parent.startdate
        stepdelta = self.parent.stepdelta
        rowdelta = steps_per_row * stepdelta

        # determine number of steps to plot
        if numsteps is None:
            numsteps = self.parent.steps[vtype][-1]

        # determine the number of row subplots
        numrows = int(math.ceil(numsteps / float(steps_per_row)))
        
        # determine line colors based on normal map
        dem_band = self.parent.init['z']
        dx, dy, dz = dem_band.calculateSurfaceNormals()
        dem = None

        # normalize to 0 - 1 interval
        r = (dx + 1) / 2
        g = (dy + 1) / 2
        b = (dz + 1) / 2

        assert np.min(r) >= 0.0
        assert np.max(r) <= 1.0
        assert np.min(b) >= 0.0
        assert np.max(b) <= 1.0
        assert np.min(g) >= 0.0
        assert np.max(g) <= 1.0
        assert line_alpha > 0.0
        assert line_alpha <= 1.0

        colors = []
        for j in xrange(ny):
            for i in xrange(nx):
                colors.append((r[j,i], g[j,i], b[j,i], line_alpha))

        # read the data from hd5 datasource
        data2d = self.read2dData()

        if mask is not None and apply_mask:
            assert data2d.shape[0] == np.prod(mask.shape)

        # determine figure size in inches
        if figsize is None:
            figsize = (20, 4*numrows)

        # initialize figure
        fig = plt.figure(figsize=figsize, dpi=dpi)

        # adjust subplot margins
        if subplots_adjust_kwargs is None:
            bottom = 0.5/figsize[1]
            top = 1.0 - bottom
            subplots_adjust_kwargs = dict(left=0.03, right=0.97,
                                          bottom=bottom, top=top)
        plt.subplots_adjust(**subplots_adjust_kwargs)

        # resample spatial locations
        n = len(colors)
        assert cell_sample_percentage > 0.0
        assert cell_sample_percentage <= 1.0

        if mask is not None and apply_mask:
            indx = np.where(mask.flatten()==1)[0]

            assert len(indx) > 0
            if cell_sample_percentage < 1.0:
                k = int(round(cell_sample_percentage * len(indx)))
                sample = random.sample(indx, k)
            else:
                sample = indx
        else:
            if cell_sample_percentage < 1.0:
                k = int(round(cell_sample_percentage * n))
                sample = random.sample(range(n), k)
            else:
                sample = range(n)
            
        # plot data subplot by subplot, trend by trend
        for r in xrange(numrows):
            if verbose:
                print('building row %i of %i...' % (r + 1, numrows))
            
            plt.subplot(numrows, 1, r+1)

            # the precipitation data can be sparse (missing frames or
            # entire rows of data) so most of the complexity here
            # is to deal with those complications
            
            t0 = startdate + rowdelta * r  # datetime.datetime object
            tend = t0 + rowdelta           # datetime.datetime object

            # datetimes list is synced to the ProcessVariable data
            j0, jend = findDateTimeIndxs(datetimes, t0, tend)

            # if pv contains data for this row
            if j0 is not None and jend is not None:
                x = self.datetimes[j0:jend]

            # for each cell that should be plotted
            for i in sample:
                
                # if pv contains data for this row
                if j0 is not None and jend is not None:
                    y = data2d[i,j0:jend]
                    color = colors[i]
                    if scatter:
                        plt.scatter(x, y, marker='.', color=color)
                    else:
                        plt.plot(x, y, marker='.', color=color)

                # make the limits consistent
                plt.xlim([t0, tend])
                
                if ylim is not None:
                    plt.ylim(ylim)

        # save figure
        if verbose:
            print('saving figure...')
        plt.savefig(dst_name)

        # close figure
        plt.close()

    def __str__(self):
        return '{0.vtype}/{0.band}\n' \
               'Hd5 built: {1}\n' \
               'shape: {0.shape}\n' \
               'Minimum: {0.min}\n' \
               'Maximum: {0.max}\n' \
               'Average: {0.avg}\n' \
               'Stdev:   {0.std}\n' \
               .format(self, self.parent.hasHd5())

class ISNOBAL:
    """
    """
    def __init__(self, input_dir, ppt_dir, output_dir,
                 init_fn, startdate, stepdelta=1.0,
                 mask_fn=None, hdf5_fname=None,
                 max_z_s_0=0.25, z_u=5.0, z_T=5.0):
        
        self.input_dir = input_dir
        self.ppt_dir = ppt_dir
        self.output_dir = output_dir
        self.hdf5_fname = hdf5_fname
        self.startdate = startdate
        if isinstance(stepdelta, timedelta):
            self.stepdelta = stepdelta
        else:
            self.stepdelta = timedelta(hours=stepdelta)
        
        self.max_z_s_0 = max_z_s_0
        self.z_u = z_u
        self.z_T = z_T

        self.fns = {}
        self.fns[IPW_IN] = glob(joinpath(input_dir, 'in.*'))
        self.fns[IPW_PPT] = glob(joinpath(ppt_dir, 'ppt*'))
        self.fns[IPW_SNOW] = glob(joinpath(output_dir, 'snow.*'))
        self.fns[IPW_EM] = glob(joinpath(output_dir, 'em.*'))

        ipw = IPW(self.fns[IPW_IN][0])
        xsize = self.xsize = ipw.nlines
        ysize = self.ysize = ipw.nsamps
        ipw = None

        self.steps = {}
        for vtype in self.fns:
            steps = np.array(map(identifyStep, self.fns[vtype]),
                             dtype=np.uint32)
            fns = self.fns[vtype]

            # Here we re-sort based on the identified steps glob sorts
            # alphabetically, if the files don't 0 pad the steps these
            # lists won't be in the right order
            self.steps[vtype], self.fns[vtype] = \
                zip(*sorted(zip(steps, fns), key=lambda tup: tup[0]))

        self.pvs = {}
        for vtype in self.fns:
            self.pvs[vtype] = \
                OrderedDict([(band, ProcessVariable(band, self))
                             for band in varlistManager[vtype]])

        # read mask if provided
        self.mask_fn = mask_fn
        if mask_fn is not None:
            _ipw = IPW(mask_fn)
            self.mask = _ipw['mask'].data
        else:
            self.mask = None
            
        # read dem if provided
        self.init_fn = init_fn
        if init_fn is not None:
            self.init = IPW(init_fn)
        else:
            self.init = None

    def datetime(self, step):
        return self.starttime + self.stepdelta * step
    
    def __iter__(self):
        d = {}
        for k in self.fns:
            d[k] = OrderedDict(zip(self.steps[k], self.fns[k]))

        s0 = np.min([np.min(L) for L in self.steps.values()])
        send = np.max([np.max(L) for L in self.steps.values()])
        for step in xrange(s0, send+1):
            yield step, dict((k, d[k].get(step, None)) for k in self.fns)
            
    def calculateSummaryStats(self):
        for vtype in self.fns:
            for band in self.pvs[vtype]:
                self.pvs[vtype][band].calculateSummaryStats()
                print(self.pvs[vtype][band])

    def __getitem__(self, path):
        grp, band = os.path.split(path)
        return self.pvs[grp][band]

    def _packgrp(self, root, grp, nbands=None, verbose=True):
        fns = self.fns[grp]
        varlist = varlistManager[grp]

        assert len(fns) > 0

        ipw0 = IPW(fns[0])
        shape = ( ipw0.nlines * ipw0.nsamps,
                  len(fns),
                  (nbands, ipw0.nbands)[nbands is None] )
        
        data = np.zeros(shape, np.float32)

        for i, fn in enumerate(fns):
            if verbose and not i % 100:
                print('.', end='')
                
            ipw = IPW(fn)
            for j, b in enumerate(ipw.bands):
                assert varlist[j] == b.name, print(varlist[j], b.name)
                data[:, i, j] = b.data.flatten()

        root.create_group(grp)
        for i, key in enumerate(varlist):
            root[grp].create_dataset(key, data=data[:, :, i])

    def packToHd5(self, hdf5_fname=None, verbose=True):
        """
        packToHd5(in_dir[, out_dir][, fname=None])

        Packs input and output data into an hdf5 container

        Parameters
        in_dir : string
            path to file containing input IPW files

        out_dir : string
            path to file containing output IPW files

        """
        if hdf5_fname is None:
            hdf5_fname = 'insnobal_data.hd5'

        in_dir = self.input_dir
        ppt_dir = self.ppt_dir
        out_dir = self.output_dir

        if not os.path.isdir(in_dir):
            raise Exception('in_dir should be a directory')

        if out_dir is None:
            out_dir = in_dir

        if not os.path.isdir(out_dir):
            raise Exception('out_dir should be a directory')

        root = h5py.File(hdf5_fname, 'w')

        # Process in_db files
        # Some of the input files have 5 bands, some have 6
        for vtype in self.fns:
            if verbose:
                print('\nPacking %s' % vtype)
            self._packgrp(root, vtype, verbose=verbose)

        root.close()
        self.hdf5_fname = hdf5_fname

    def hasHd5(self):
        hdf5_fname = self.hdf5_fname

        if hdf5_fname is None:
            return False

        return os.path.exists(hdf5_fname)

if __name__ == '__main__':

    data_dir = 'D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL'
    
    input_dir = joinpath(data_dir, 'input.dp')
    ppt_dir = joinpath(data_dir, 'ppt_images_dist')
    output_dir = joinpath(data_dir, 'output_500_05_3_1.1_5')
    init_fn = joinpath(data_dir, 'init.ipw')
    mask_fn = joinpath(data_dir, 'tl2p5mask.ipw')
    startdate = datetime(2010, 10, 1)

    hdf5_fname = joinpath(data_dir, 'dryCreek.hd5')
    
    run = ISNOBAL(input_dir, ppt_dir, output_dir, init_fn, startdate,
                  mask_fn=mask_fn, hdf5_fname=hdf5_fname)


    pv = run['em/M']
    pv.calculateSummaryStats()

    print(pv.min, pv.max)
    
    dst_fname = 'D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL\AspectEnsemble__M.png'
    pv.aspectEnsemblePlot(dst_fname, steps_per_row=24*14,
                          ylim=[pv.min, pv.max], scatter=True)


##    for step, fns_dict in run:
##        print(step)
##        pp(fns_dict)
##        raw_input()

    
##    fn = run.fns[IPW_SNOW][1500]
##    ipw = IPW(fn)
##    print(fn, ipw.name_dict)
##
###    dst_fname = joinpath(data_dir, 'snow.1500.swe.tif')
###    swe = ipw.generateSWE(dst_fname)
##    
##    fn = run.fns[IPW_IN][9]
##    ipw = IPW(fn)
##    
##    dst_fname = joinpath(data_dir, 'in.0010.S_n.tif')
##    ipw.generateS_n(dst_fname)
##
##    from bitstring import BitArray
##    from random import shuffle
##    
##    binwords = set()
##    samples = []
##    fns = list(run.fns[IPW_PPT])
##
##    shuffle(fns)
##    for fn in fns[:100]:
##        
##        ipw = IPW(fn, rescale=False)
##        data = ipw['m_pp'].data
##        print(fn, data.dtype)
##
##        string = data.tostring()
##
##        print(len(string))
##
##        for i in xrange(0, len(string)-2, 2):
##            c = BitArray(bytes=string[i:i+2])
##            binstr = c.bin
##            binwords.add(binstr)
##
##            c = BitArray(bin=binstr[:12])
##            samples.append(c.uint)
##
##    truncated_reversed = [] 
##    for binstr in binwords:
##        truncated_reversed.append(binstr[:12])
##
##    for binstr in truncated_reversed:
##        c = BitArray(bin=binstr)
##        print(c.bin, c.uint)
##
##    plt.figure()
##    plt.hist(samples, bins=512)
##    plt.show()
##    plt.figure()
##    plt.imshow(m2)
##    plt.colorbar()
##    plt.show()
    
#    run.packToHd5(hdf5_fname)

##    pv = run['ppt/m_pp']
##    pv.calculateSummaryStats()
##
##    print(pv.min, pv.max)
##    
##    dst_fname = 'D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL\AspectEnsemble__m_pp.png'
##    pv.aspectEnsemblePlot(dst_fname, steps_per_row=24*14, ylim=[0, 15], scatter=True)

    
##    run.calculateSummaryStats()

##    
##    pv = run['in/S_n']
##    pv.calculateStats()
##    working_dir = joinpath('D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL\processed', 'S_n')
##    clean(working_dir, ispath=True)
##    
##    print(pv.min, pv.avg, pv.max, pv.std)
##
##    i = 0
##    for band in pv:
##        basename = os.path.basename(band.parent.fname)
##        dst_name = joinpath(working_dir, '%s.%s.tif' % (basename, band.name))
##        band.colorize(dst_name, **rcParams['in/S_n'])
##
##        i += 1
##        if i == 196:
##            break

##
##    band = 'I_lw'
##    pv = run['in/%s' % band]
##    
##    working_dir = joinpath('D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL\processed', band)
##    clean(working_dir, ispath=True)
##    
##    print(pv.min, pv.avg, pv.max, pv.std)
##
##    transform = lambda x: (x - pv.avg) / pv.std
##    ymag = max(abs(transform(pv.min)), abs(transform(pv.max)))

##    global mu, sigma
##    mu = pv.avg
##    sigma = pv.std
##    
##    print(rcParams['in/%s' % band])
##    
##    i = 0
##    for band_obj in pv:
##        basename = os.path.basename(band_obj.parent.fname)
##        dst_name = joinpath(working_dir, '%s.%s.tif' % (basename, band))
##        band_obj.colorize(dst_name, **rcParams['in/%s' % band])
##
##        i += 1
##        if i == 196:
##            break
