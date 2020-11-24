import numpy as np
import neoml.MathEngine as M
import neoml.Blob as B

mathEngine = M.CpuMathEngine(1)

data = np.ones( (500, 11, 6), dtype=np.float32 )

shape = ( 4 )
blob = B.data_blob( mathEngine, 500, 11, 6, "int32", data )

blob_data = blob.data()

print( blob.shape() )
#print( blob_data )
print( "The end" )
	

 