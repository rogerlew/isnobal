#!/usr/local/bin/python
"""
Example of how to modify the observed Kormos data's temperature by a values and
save the modified results

same thing as:
https://github.com/tri-state-epscor/adaptors/blob/master/examples/modify_ipw_temps.py
"""
from __future__ import print_function

import sys
import os

from isnobal import IPW


def clean(path, ispath=False):
    """
    cleans path by deleting path if it exists and making directory if
    path is a directory or ispath is True
    """
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            time.sleep(1)
            os.mkdir(path)
        else:
            os.remove(path)

    elif ispath:
            os.mkdir(path)

if __name__ == "__main__":
    try:
        amount = float(sys.argv[1])
        inputFile = sys.argv[2]
    except:
        print("Expecting amount as scalar and inputFile as cmd args")
    
    basename = os.path.basename(inputFile)

    output_dir = "data/inputsP%f/" % amount
    clean(output_dir)

    ipw = IPW(inputFile)
    ipw['T_a'] += amount
    ipw.write(os.path.join(output_dir, basename))
