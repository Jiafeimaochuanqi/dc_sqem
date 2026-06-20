from pyigl.unique_rows import *
from pyigl.sort import *
def unique_simplices(F):
    sortF,_ = sort(F, dim=2, ascending=True)
    FF, IA, IC = unique_rows(sortF)
    FF = F[IA]

    return FF, IA, IC