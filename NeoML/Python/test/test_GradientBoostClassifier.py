import numpy as np
from scipy.sparse import csr_matrix
import array as ar
import neoml.GradientBoost as G
import svml

print("Reading news20.train.svml")

X, y = svml.read( "data/news20.train.svml" )
weight = np.ones( len(y), dtype=float )

print("Training")

boost = G.GradientBoostClassifier( loss ='binomial', iteration_count=10 )
model = boost.train( X, y, weight )

print("Reading news20.test.svml")

X_test, y_test = svml.read( "data/news20.test.svml" )

print("Testing")

correct = sum( 1 for y, probs in zip( y_test, model.classify( X_test ) ) if y == np.argmax( probs ) )
print( float( correct ) / len( y_test ) )

print("Done.") 
