from __future__ import print_function

"""
run nosetests from module root
"""
from hashlib import sha224
import os
import time
import unittest
from pprint import pprint
import numpy as np

from numpy.testing import assert_array_equal, \
                          assert_array_almost_equal

from isnobal import IPW


def assert_bands_equal(b1, b2):
    assert b1.nlines == b2.nlines
    assert b1.nsamps == b2.nsamps 

    # basic_image
    assert b1.name == b2.name
    assert b1.bytes == b2.bytes
    assert b1.fmt == b2.fmt
    assert b1.bits == b2.bits

    # geo
    assert b1.bline == b2.bline
    assert b1.bsamp == b2.bsamp
    assert b1.dline == b2.dline
    assert b1.dsamp == b2.dsamp
    assert b1.geounits == b2.geounits
    assert b1.coord_sys_ID == b2.coord_sys_ID
    assert b1.geotransform == b2.geotransform

    # lq
    assert b1.x0 == b2.x0
    assert b1.xend == b2.xend
    assert b1.y0 == b2.y0
    assert b1.yend == b2.yend
    assert b1.units == b2.units

    assert b1.data.shape == b2.data.shape

#    nsamp, nline = b1.data.shape
#    
#    for i in xrange(nsamp):
#        for j in xrange(nline):
#            assert b1.data[i,j] == b2.data[i,j], \
#                   print(i,j, b1.data[i,j], b2.data[i,j])
            
    assert_array_almost_equal(b1.data, b2.data)
    
    return True

class Test_readIPW(unittest.TestCase):
    
    def test_read(self):
        ipw = IPW('testIPWs/in.0051', rescale=False)
        """
        with open('validation/in.0051.validation.txt', 'wb') as f:
            for L in ipw.bands[0].data:
                f.write('\t'.join(map(str, L)))
                f.write('\r\n')
        """
        with open('validation/in.0051.validation.txt') as f:
            for d, L in zip(ipw.bands[0].data, f.readlines()):
                x =  np.array(map(int, L.split('\t')))
                assert_array_equal(d, x)
                            
    def test_read_scale(self):
        ipw = IPW('testIPWs/in.0051')
        """
        with open('validation/in.0051.validation2.txt', 'wb') as f:
            for L in ipw.bands[0].data:
                f.write('\t'.join(map(str, L)))
                f.write('\r\n')
        """
        with open('validation/in.0051.validation2.txt') as f:
            for d, L in zip(ipw.bands[0].data, f.readlines()):
                x = np.array(map(float, L.split('\t')))
                assert_array_almost_equal(d, x, 3)

    def test_read_gtiff(self):
        tmp_fn = 'tmp/in.0051'
        
        ipw = IPW('testIPWs/in.0051', rescale=False)
        ipw.translate(tmp_fn)

        ipw2 = IPW(tmp_fn + '.tif', rescale=False)
        
        for b1, b2 in zip(ipw, ipw2):
            assert_bands_equal(b1, b2)

        os.remove(tmp_fn + '.tif')
        time.sleep(0.5)
        
    def test_read_gtiff_scale(self):
        tmp_fn = 'tmp/in.0051'
        
        ipw = IPW('testIPWs/in.0051')
        ipw.translate(tmp_fn)

        ipw2 = IPW(tmp_fn + '.tif')
        
        for b1, b2 in zip(ipw, ipw2):
            assert_bands_equal(b1, b2)

        os.remove(tmp_fn + '.tif')
        time.sleep(0.5)
        
class Test_writeIPW(unittest.TestCase):
    
    def test_write(self):
        tmp_fn = 'tmp/in.0051'
        
        ipw = IPW('testIPWs/in.0051', rescale=False)
        ipw.write(tmp_fn)

        ipw2 = IPW(tmp_fn, rescale=False)
        
        for b1, b2 in zip(ipw, ipw2):
            assert_bands_equal(b1, b2)

        os.remove(tmp_fn)
        time.sleep(0.5)

        
    def test_write_scale(self):
        tmp_fn = 'tmp/in.0051'
        
        ipw = IPW('testIPWs/in.0051')
        ipw.write(tmp_fn)

        ipw2 = IPW(tmp_fn)
        
        for b1, b2 in zip(ipw, ipw2):
            assert_bands_equal(b1, b2)

        os.remove(tmp_fn)
        time.sleep(0.5)
         
class Test_translate(unittest.TestCase):               
    def test_translate001(self):
        ipw = IPW('testIPWs/in.0051', rescale=False)
        ipw.translate('tmp/in.0051', multi=True)

        fns = [ 'tmp/in.0051.00.tif',
                'tmp/in.0051.01.tif',
                'tmp/in.0051.02.tif',
                'tmp/in.0051.03.tif',
                'tmp/in.0051.04.tif' ]
        """
        f = open('hash.txt', 'wb')
        for fn in fns:
            f.write(sha224(open(fn).read()).hexdigest())
            f.write('\r\n')
        """
        digests = [ '3275154565f7b40b56f1567f08d501c50480150b506ac60a769d382a',
                    '6e5b2472b45ddac9d8a22c5c21a5e59ca611ab2ff734dfe5881cf86d',
                    'dbb2728f4faf960472b6b4200b8ea939d80d59da3be482dee9ba2cfa',
                    'dbb2728f4faf960472b6b4200b8ea939d80d59da3be482dee9ba2cfa',
                    '6e5b2472b45ddac9d8a22c5c21a5e59ca611ab2ff734dfe5881cf86d' ]
        
        for fn in fns:
            assert os.path.exists( fn )

        for i, fn in enumerate(fns):
            hd = sha224(open(fn).read()).hexdigest()
            assert hd == digests[i]
            
        for fn in fns:
            os.remove( fn )
          
    def test_translate003(self):
        ipw = IPW('testIPWs/in.0051')
        ipw.translate('tmp/in.0051', multi=False)

        fns = [ 'tmp/in.0051.tif']

        """        
        f = open('hash.txt', 'wb')
        for fn in fns:
            f.write(sha224(open(fn).read()).hexdigest())
            f.write('\r\n')
        """
        
        digests = [ '0b71f82644cba687f620fbf593f1c37a131473a8a356d0903c7798ea' ]
        
        for fn in fns:
            assert os.path.exists( fn )

        for i, fn in enumerate(fns):
            hd = sha224(open(fn).read()).hexdigest()
            assert hd == digests[i]

        for fn in fns:
            os.remove( fn )
            
    def test_translate004(self):
        ipw = IPW('testIPWs/in.0051')
        ipw.translate('tmp/in.0051', multi=False, writebands=[1,3,4])

        fns = [ 'tmp/in.0051.tif']
        """
        f = open('tests/hash.txt', 'wb')
        for fn in fns:
            f.write(sha224(open(fn).read()).hexdigest())
            f.write('\r\n')
        """
        digests = [ '91356e8d5d103f2637c5cd24b30dad683cba2254393acf4e107db8ce' ]
         
        for fn in fns:
            assert os.path.exists( fn )

        for i, fn in enumerate(fns):
            hd = sha224(open(fn).read()).hexdigest()
            assert hd == digests[i]

        for fn in fns:
            os.remove( fn )

class Test_hd5(unittest.TestCase):               
    def test_packToHD5(self):
        fname = 'tmp/data.hd5'
#        packToHd5(os.path.join('tests', 'testSet'), fname=fname)

        time.sleep(1)
        
        os.remove( fname )
        
def suite():
    return unittest.TestSuite((
            unittest.makeSuite(Test_readIPW),
            unittest.makeSuite(Test_writeIPW),
            unittest.makeSuite(Test_translate),
#            unittest.makeSuite(Test_hd5)
                              ))

if __name__ == "__main__":

    
    # run tests
    runner = unittest.TextTestRunner()
    runner.run(suite())
