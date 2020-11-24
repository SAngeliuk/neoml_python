import sys
import os
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import array as ar
import neoml


mathEngine = neoml.MathEngine.CpuMathEngine(1)

dnn = neoml.Dnn.Dnn( mathEngine, path='data/MobileNetV2Cifar10.cnnarch')

blob = neoml.Blob.image2d( mathEngine, 1, 1, 32, 32, 3 )

print( type( blob ) )

print( dnn.layers )

                                 
#except:
#	print("Unexpected error:", sys.exc_info()[0])
 