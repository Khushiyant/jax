import cppyy, numba, warnings
import cppyy.numba_ext
import numpy as np
from cppyy.gbl.std import vector

cppyy.load_library("/usr/lib/x86_64-linux-gnu/libiomp5.so")

cppyy.include("/workspaces/jax/jax/_src/cppyy/mat/matmul.cpp")

def mul_njit(d, shape1, shape2):
  d.multiplyMatrices(shape1, shape2)

def std_vecmul(qy, db):
  shape_qy = vector[int](qy.shape)
  shape_db = vector[int](db.shape)

  res = vector["double"](np.zeros(shape_qy[0] * shape_db[1]))
  qy = vector["double"](qy.flatten())
  db = vector["double"](db.flatten())
  d = cppyy.gbl.MatrixDot(qy, db, res)
  mul_njit(d, shape1=shape_qy, shape2=shape_db)
  return np.array(d.result).reshape(shape_qy[0], shape_db[1])