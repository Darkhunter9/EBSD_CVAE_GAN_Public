"""
Copyright (c) 2019-2022, Zihao Ding/Carnegie Mellon University
All rights reserved.
********************************************************************

Project: eu2qu.py

MODULE: util

Author: Zihao Ding, Carnegie Mellon University

Brief:
-------------
Refer to source code in https://github.com/marcdegraef/3Drotations
convert Euler angles to quaternions


Date:
-------------
2022/03/17 ZD 1.0 public version
"""

from math import sin, cos
import numpy as np

def eu2qu(eu):
    '''
    input: euler angles, array-like (3,)
    output: quaternions, array-like (4,)
    default value of eps = 1
    '''

    eps = 1

    sigma = 0.5 * (eu[0] + eu[2])
    delta = 0.5 * (eu[0] - eu[2])
    c = cos(eu[1]/2)
    s = sin(eu[1]/2)

    q0 = c * cos(sigma)

    if q0 >= 0:
        q = np.array([c*cos(sigma), -eps*s*cos(delta), -eps*s*sin(delta), -eps*c*sin(sigma)], dtype=float)
    else:
        q = np.array([-c*cos(sigma), eps*s*cos(delta), eps*s*sin(delta), eps*c*sin(sigma)], dtype=float)

    # set values very close to 0 as 0
    # thr = 10**(-10)
    # q[np.where(np.abs(q)<thr)] = 0.

    return q