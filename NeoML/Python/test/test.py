import sys
import os
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
import array as ar
import neoml

#import numpy as np
#import math

this_dir = os.path.dirname(os.path.realpath(__file__))

# path to the location of the binding module
sys.path.append('f:/Work/Android2018_2/ResultPython/lib')

# name of the binding module: pybind11_binding
import pickle

print("Reading Iris.txt")
file = open("Iris.txt", "r")

#################################

row = np.array([0, 1, 2, 0])
col = np.array([0, 1, 1, 0])
data = np.array([1, 2, 4, 8])
mat = csr_matrix((data, (row, col)), shape=(3, 3))

#print( mat.toarray() )

#print( mat.indptr )

###################################################

weight1 = np.ones( row.size, dtype=float )

data = []
col = []
row = ar.array( 'i', [ 0 ] )
y = []
rowCount = 0;

for line in file:
	cur = line.split(",")
	data.append( float(cur[0]) )
	data.append( float(cur[1]) )
	data.append( float(cur[2]) )
	data.append( float(cur[3]) )
	col.append( 0 )
	col.append( 1 )
	col.append( 2 )
	col.append( 3 )
	rowCount += 4
	row.append( rowCount )

	y.append( int(cur[4]) )

xxx = ( 1, 2, 3, 4, 5 )

ar = ar.array( 'f', xxx )


print( type( xxx ) )
print( type( ar ) )


X = csr_matrix( ( np.array( data, np.float32 ), np.array( col, np.int32 ), row ), shape=( int( rowCount / 4 ), 4) )
#X = csr_matrix( ( np.array( data, np.float32 ), np.array( col, np.int32 ), np.array( row, np.int32 ) ), shape=( int( rowCount / 4 ), 4) )

t = [[ 1, 2, 3 ], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

try:

	print( "Begin external..." )

#	boost = G.GradientBoostClassifier( loss ='binomial', iteration_count=1 )
#	boost._train_ndarray( X.data )

	print( "Begin external..." )


	print( "Train..." )
	
#	model = boost.train( X, y )

#	model.store( 'gb.carchive' )

#	binary_file = open('model.bin', mode='wb')
#	my_pickled_model = pickle.dump(model, binary_file)
#	binary_file.close()


#	loaded_model = pickle.load(open('model.bin', 'rb'))

#	print( type( loaded_model ) )


#	print( "Test..." )
#	test = [ [5.3,3.7,1.5,0.2], [5.0,3.3,1.4,0.2], [7.0,3.2,4.7,1.4], [6.4,3.2,4.5,1.5] ]

#	print( type(loaded_model) )
#	res = loaded_model.classify( test )
#	print( type(type(res)) )
#	print( res )
	
#	model2 = G.GradientBoostClassificationModel( 'gb.carchive' )

#	res = model2.classify( test )
#	print( type(type(res)) )
#	print( res )

#------------------------------------------------


#------------------------------------------------

	mathEngine = neoml.MathEngine.CpuMathEngine(1)

	dnn = neoml.Dnn.Dnn( mathEngine, 'MobileNetV2Cifar10.cnnarch')

	blob = neoml.Blob.image2d( mathEngine, 32, 32, 3 )

	print( type( blob ) )
	print( blob._get_height() )


	x1 = np.ones( (32, 32, 1, 3), dtype=float )
	xlist1 = []
	xlist1.append( x1 )
	y1 = dnn.run( xlist1 )

#	print( type( y1 ) )
	print( y1 )
	

except RuntimeError as err:
    print("RuntimeError error: {0}".format(err))
except MemoryError as err:
    print("MemoryError error: {0}".format(err))
#except:
#	print("Unexpected error:", sys.exc_info()[0])
 