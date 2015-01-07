from __future__ import print_function

# Copyright (c) 2014, Roger Lew (rogerlew.gmail.com)
#
# The project described was supported by NSF award number IIA-1301792 
# from the NSF Idaho EPSCoR Program and by the National Science Foundation.

"""
Parallel batch process .daq files to .hdf5 or .mat

By default warnings are suppressed but can be attained with the -d or --debug

Sample usage
------------
D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL>C:\Python27x64\python.exe isnobal\scripts\batch_translate.py input.dp processedTiffs5 -d -n 8
src_path: input.dp
dst_path: processedTiffs5
drivername: GTiff
writebands: None
epsg: 32611
multi: False
numcpu: 8
debug: True

Creating dst_path directory...

Starting multiprocessing pool with 8 workers

Found 8759 IPW files to translate

Converting IPWs with 8 cpus (this may take awhile)...

Batch processing completed.

--------------------------------------------------------------------
Conversion Summary
--------------------------------------------------------------------
Total elapsed time: 13.6 s
Data converted: 1,694.459 MB
Data throughput: 124.8 MB/s
--------------------------------------------------------------------

D:\ownCloud\documents\geo_data\DryCreek\iSNOBAL>
"""

from ast import literal_eval
import argparse
from collections import namedtuple
import os
from glob import glob
import time
import multiprocessing
import warnings

from isnobal import IPW

def ipwToTif(tupledArgs):
    src_fname, dst_fname, writebands, drivername, epsg, multi = tupledArgs
    ipw = IPW(src_fname, epsg=epsg)
    ipw.translate(dst_fname,  writebands=writebands,
                  drivername=drivername, multi=multi)
    return 1
  
if __name__ == '__main__':

    # setup argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('src_path', type=str,   
        help='Path to dir containing IPWs')

    parser.add_argument('dst_path', type=str,   
        help='Path to write rasters')
    
    parser.add_argument('-n', '--numcpu',   type=int, 
        help='Number of cpus in pool    (1)')
    
    parser.add_argument('-o', '--outtype',   
        help='Output type               ([GTiff], )')

    parser.add_argument('-b', '--bands', 
        help='List of comma separated bands to translate ("")')
        
    parser.add_argument('-e', '--epsg', 
        help='EPSG of source IPWs       (32611)')
    
    parser.add_argument('-m', '--multi',  
        help='Multitranslate will write each band to its own dataset',
        action='store_true')
    
    parser.add_argument('-d', '--debug',  
        help='Print the return codes',
        action='store_true')

    # parse arguments
    args = parser.parse_args()
    src_path = args.src_path
    dst_path = args.dst_path
    epsg = 32611
    if args.epsg is not None:
        epsg = int(args.epsg)
    numcpu = (args.numcpu, 1)[args.numcpu is None]
    drivername = (args.outtype, 'GTiff')[args.outtype is None]
    writebands = (args.bands, None)[args.bands is None]
    if writebands is not None:
        # literal eval is safer than eval
        writebands = literal_eval('[' + writebands + ']')

    multi = args.multi
    debug = args.debug

    if debug:
        print('src_path:', src_path)
        print('dst_path:', dst_path)
        print('drivername:', drivername)
        print('writebands:', writebands)
        print('epsg:',epsg)
        print('multi:', multi)
        print('numcpu:', numcpu)
        print('debug:', debug)
    
    if not os.path.exists(dst_path):
        print(('', '\nCreating dst_path directory...')[debug])
        os.makedirs(dst_path)
        
    # parallel worker pool
    print(('', '\nStarting multiprocessing pool with %i workers'%numcpu)[debug])
    pool = multiprocessing.Pool(numcpu)

    fns = glob(os.path.join(src_path, 'in.*'))
    fns.extend(glob(os.path.join(src_path, 'em.*')))
    fns.extend(glob(os.path.join(src_path, 'snow.*')))
    fouts = [os.path.join(dst_path, os.path.basename(fn)) for fn in fns]
    argslist = [ (fn, fout, writebands, drivername, epsg, multi) \
                 for fn, fout in zip(fns, fouts) ]
    print(('', '\nFound %i IPW files to translate'%len(fns))[debug])
    
    # ready to roll.
    print('\nConverting IPWs with %i cpus (this may take awhile)...'%numcpu)    
    t0 = time.time() # start global time clock 

## no multiprocessing
##
## the multiprocessing obscures some of the stacktrace so it
## pretty difficult to debug with it running
#    for arg in argslist:
#        ipwToTif(arg)            
    
    # this launches the batch processing of the daq files
    results = pool.imap(ipwToTif, argslist)

    # results is an iterator! you can only traverse it once.
    retcodes = []
    for retcode in results:
        retcodes.append(retcode)

    # close multiprocessing pool
    pool.close()
    pool.join()
    
    elapsed_time = time.time() - t0 + 1e-6 # so throughput calc doesn't bomb
                                           # when argslist is empty
                                           
    # calculate the amount of data that was converted in MB
    tot_mb = sum(os.stat(fn).st_size/(1024*1024.) for fn in fns)
    
    # provide some feedback to the user
    print('\nBatch processing completed.\n')
    print('-'*(43+13+12))
    print('Conversion Summary')
    print('-'*(43+13+12))
    print('Total elapsed time: %.1f s'%(elapsed_time))
    print('Data converted: {:,.3f} MB'.format(tot_mb))
    print('Data throughput: %.1f MB/s'%(tot_mb/elapsed_time))
    print('-'*(43+13+12))


