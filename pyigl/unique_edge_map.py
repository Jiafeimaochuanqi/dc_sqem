from pyigl.oriented_facets import *
from pyigl.unique_simplices import *
from pyigl.accumarray import *
from pyigl.cumsum import *

import numpy as np

def unique_edge_map(F,return_uE2E=False, return_uEC=False, return_uEE=False):
    if return_uE2E:
        E, uE, EMAP=unique_edge_map_base(F)
        uE2E=[[] for _ in range(uE.shape[0])]
        for e in range(EMAP.shape[0]):
            uE2E[EMAP[e]].append(e)
        return E, uE, EMAP,uE2E
    elif return_uEC and return_uEE:
        E, uE, EMAP=unique_edge_map_base(F)
        uEK = accumarray_scalar(EMAP,1)
        assert(uEK.shape[0]==uE.shape[0])
        #base offset in uEE
        uEC=cumsum(uEK,1,True).astype(np.int)
        assert(uEK.shape[0]+1 == uEC.shape[0])
        uEO = np.zeros([uE.shape[0],1],dtype=np.int)
        uEE=np.zeros([EMAP.shape[0],1],dtype=np.int)
        for e in range(EMAP.shape[0]):
            ue = EMAP[e]
            i = uEC[ue] + uEO[ue]
            uEE[i] = e
            uEO[ue] += 1
        assert np.all(uEK == uEO)
        return E, uE, EMAP,uEK,uEC,uEE
    else:
        return unique_edge_map_base(F)

def unique_edge_map_base(F):
    E=oriented_facets(F)
    uE,_,EMAP=unique_simplices(E)
    return E,uE,EMAP
