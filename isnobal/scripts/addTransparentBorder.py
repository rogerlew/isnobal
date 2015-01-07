"""
The Unity3D projector shader that we are using needs images that have a 
transparent border to avoid tiling artifacts.
"""

import sys

import numpy as np
import scipy.misc 

def addTransparentBorder(src_fname, dst_fname):
    rgba = scipy.misc.imread(src_fname)

    w, h, n = rgba.shape

    if n == 3:
        alpha = np.ones((w,h,1), dtype=np.uint8) * 255
        rgba = np.concatenate((rgba, alpha), axis=2)

    rgba[0,:,3] = 0
    rgba[w-1,:,3] = 0
    rgba[:,0,3] = 0
    rgba[:,h-1,3] = 0

    scipy.misc.imsave(dst_fname, rgba)

if __name__ == "__main__":

    try:
        src_fname = sys.argv[1]
        dst_fname = sys.argv[2]
    except:
        print("Expecting src_fname and dst_fname")
    

    
