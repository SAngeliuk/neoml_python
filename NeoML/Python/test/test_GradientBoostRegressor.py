import numpy as np
from scipy.sparse import csr_matrix
import array as ar
import neoml.GradientBoost as G
import svml

print("Reading news20bin.train.svml")

X, y = svml.read( "data/news20bin.train.svml" )
weight = np.ones( len(y), dtype=float )

print("Training")

boost = G.GradientBoostRegressor( loss='l2', iteration_count=100, learning_rate=0.3, max_depth=12, l2_reg=0, thread_count=4, subsample=1, subfeature=0.8 )
model = boost.train( X, y, weight )

print("Reading news20bin.test.svml")

X_test, y_test = svml.read( "data/news20bin.test.svml" )

print("Testing")

correct = sum( 1 for y, prob in zip( y_test, model.predict( X_test ) ) if ( y == 1 if prob >= 0.5 else y == 0 ) )
print( float( correct ) / len( y_test ) )

print("Done.") 
