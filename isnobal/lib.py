
import os
import shutil
import time

def clean(path, isdir=None):
    """
    cleans the path

    if path is a file it is removed, if path is a
    directory the directory tree is removed and the
    root directory is recreated

    Parameters
    ----------
    path : string
        path to be cleaned

    isdir : bool
        if path does not currently exist and ispath
        is true a new directory is created
    """
    
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            time.sleep(1)
            os.mkdir(path)
        else:
            os.remove(path)

    elif isdir:
            os.mkdir(path)
        
def identifyStep(fname):
    """
    identifies the simulation step from a filename

    Parameters
    ----------
    fname : string
        only the basename is used for determining step

    Returns
    -------
    step : int
        integer representing the simulation step
    """

    basename = os.path.basename(fname)

    if 'dem' in basename:
        return 0
    elif 'mask' in basename:
        return 0
    elif 'init' in basename:
        try:
            return int(''.join([a for a in basename if a in '0123456789']))
        except:
            return 0
        
    try:
        return int(''.join([a for a in basename.split('_')[1]
                            if a in '0123456789']))
    except:
        try:
            return int(basename.split('.')[1])
        except:
            warnings.warn('Could not identify step for "%s", '
                          'returning 0'% basename)
            return 0
        
