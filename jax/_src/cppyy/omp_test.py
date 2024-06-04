import time
import cppyy, numba, warnings
import cppyy.numba_ext
import os
import numpy as np
from cppyy.gbl.std import vector

cppyy.load_library("/usr/lib/x86_64-linux-gnu/libiomp5.so")

cppyy.include("/workspaces/jax/jax/_src/cppyy/mat/matmul.cpp")

cppyy.gbl.ompdemo()
