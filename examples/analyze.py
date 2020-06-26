import numpy as np
import sys
import h5py
import datetime

import embedding.orbital_assign_utils_py3 as ut



localHotSpotPos = ut.AngToBohr([-2.459512,-6.39,1.8])

print("Orbital localization stats - atomic core")
ut.stats_from_sizeDist('sizeDist-Co+C30H22-allFB-SO+V2=82.core',origin=localHotSpotPos)

print("Orbital localization stats - doubly occupied")
ut.stats_from_sizeDist('sizeDist-Co+C30H22-allFB-SO+V2=82.do',origin=localHotSpotPos)

print("Orbital localization stats - singly occupied + virtual 2 (82 virtuals)")
ut.stats_from_sizeDist('sizeDist-Co+C30H22-allFB-SO+V2=82.so',origin=localHotSpotPos)

print("Orbital localization stats - virtual 1")
ut.stats_from_sizeDist('sizeDist-Co+C30H22-allFB-SO+V2=82.v',origin=localHotSpotPos)
