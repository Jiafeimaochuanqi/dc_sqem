import numpy as np
import math
import math
import numpy as np
from decimal import Decimal, getcontext

getcontext().prec = 50

#Define return type constants.
POSITIVE = 1
NEGATIVE = -1
ZERO = 0


def determinant(a1, a2, a3, b1, b2, b3, c1, c2, c3):
    """
    Calculate the determinant of a 3x3 matrix.
    | a1 a2 a3 |
    | b1 b2 b3 |
    | c1 c2 c3 |
    """
    return (a1 * (b2 * c3 - b3 * c2) - 
            a2 * (b1 * c3 - b3 * c1) + 
            a3 * (b1 * c2 - b2 * c1))

def sse2minmax(maxx, maxy, maxz):
    """
    Simulate the minmax operation of SSE2 instructions to sort three values, so that maxx <= maxy <= maxz.
    """
    arr = [maxx, maxy, maxz]
    arr.sort()
    maxx, maxy, maxz = arr
    return maxx, maxy, maxz

def orientation_3(p, q, r, s, base_operator=None):
    try:
        px, py, pz = float(p[0]), float(p[1]), float(p[2])
        qx, qy, qz = float(q[0]), float(q[1]), float(q[2])
        rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
        sx, sy, sz = float(s[0]), float(s[1]), float(s[2])
    except (OverflowError, ValueError):
        if base_operator:
            return base_operator(p, q, r, s)
        else:
            return exact_orientation_3(p, q, r, s)

    pqx = qx - px
    pqy = qy - py
    pqz = qz - pz
    
    prx = rx - px
    pry = ry - py
    prz = rz - pz
    
    psx = sx - px
    psy = sy - py
    psz = sz - pz
    
    maxx = abs(pqx)
    maxy = abs(pqy)
    maxz = abs(pqz)
    
    aprx = abs(prx)
    apsx = abs(psx)
    
    apry = abs(pry)
    apsy = abs(psy)
    
    aprz = abs(prz)
    apsz = abs(psz)
    
    maxx = max(maxx, aprx, apsx)
    maxy = max(maxy, apry, apsy)
    maxz = max(maxz, aprz, apsz)

    det = determinant(pqx, pqy, pqz, prx, pry, prz, psx, psy, psz)

    eps = 5.1107127829973299e-15 * maxx * maxy * maxz
    
    maxx, maxy, maxz = sse2minmax(maxx, maxy, maxz)
    
    if maxx < 1e-97:
        if maxx == 0:
            return ZERO
    
    if maxz < 1e102:
        if det > eps:
            return POSITIVE
        if det < -eps:
            return NEGATIVE
    
    if base_operator:
        return base_operator(p, q, r, s)
    else:
        return exact_orientation_3(p, q, r, s)

def exact_orientation_3(p, q, r, s):
    px, py, pz = Decimal(p[0]), Decimal(p[1]), Decimal(p[2])
    qx, qy, qz = Decimal(q[0]), Decimal(q[1]), Decimal(q[2])
    rx, ry, rz = Decimal(r[0]), Decimal(r[1]), Decimal(r[2])
    sx, sy, sz = Decimal(s[0]), Decimal(s[1]), Decimal(s[2])
    pqx = qx - px
    pqy = qy - py
    pqz = qz - pz
    
    prx = rx - px
    pry = ry - py
    prz = rz - pz
    
    psx = sx - px
    psy = sy - py
    psz = sz - pz

    det = (pqx * (pry * psz - prz * psy) - 
           pqy * (prx * psz - prz * psx) + 
           pqz * (prx * psy - pry * psx))
    
    if det > 0:
        return POSITIVE
    elif det < 0:
        return NEGATIVE
    else:
        return ZERO

def compute_approximate_dihedral_angle(a, b, c, d):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    
    # compute vector ab, ac, ad
    ab = b - a
    ac = c - a
    ad = d - a
    
    '''
                 n1
             ↑
             |
        c    | 
       / \   |
      /   \  |
     a-----b--------> ab
      \   /
       \ /
        d
        |
        ↓
       n2
    '''
    ab_ad_cross = np.cross(ab, ad)  # ab × ad
    ab_ac_cross = np.cross(ab, ac)  # ab × ac
    
    x = np.dot(ab_ac_cross, ab_ad_cross)
    
    l_ab = np.linalg.norm(ab)
    
    y = l_ab * np.dot(ac, ab_ad_cross)

    angle_rad = math.atan2(y, x)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg


    