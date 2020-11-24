import numpy as np
from scipy.sparse import csr_matrix
import array as ar
import neoml.Linear as L
import svml

print("Reading news20bin.train.svml")

X, y = svml.read( "data/news20bin.train.svml" )
weight = np.ones( len(y), dtype=float )

print("Training")

linear = L.LinearRegressor( loss ='l2' )
model = linear.train( X, y, weight )

print("Reading news20bin.test.svml")

X_test, y_test = svml.read( "data/news20bin.test.svml" )

print("Testing")

answer = model.predict( X_test )

correct = 0

correct = sum( 1 for y, prob in zip( y_test, model.predict( X_test ) ) if ( y == 1 if prob >= 0.5 else y == 0 ) )
print( float( correct ) / len( y_test ) )

print("Done.") 
