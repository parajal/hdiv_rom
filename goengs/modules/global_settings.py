# some global settings for model templates

realcompile = False

import ngsolve

ngsolve.ngsglobals.msg_level = 3
ngsolve.SetHeapSize(100*1000*1000)


### THE GLOBAL SETTINGS CAN BE OVERWRITTEN BY ADDING 
# local_settings.py to your pythonpath
try:
    from local_settings import *
except:
    pass