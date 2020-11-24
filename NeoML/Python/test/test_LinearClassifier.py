import numpy as np
import neoml.Linear as L
import svml

print("Reading news20bin.train.svml")

X, y = svml.read( "data/news20bin.train.svml" )
weight = np.ones( len(y), dtype=float )

print("Training")

linear = L.LinearClassifier( loss='squared_hinge', l1_reg=0.05 )
model = linear.train( X, y, weight )

print("Reading news20bin.test.svml")

X_test, y_test = svml.read( "data/news20bin.test.svml" )

print("Testing")

correct = sum( 1 for y, probs in zip( y_test, model.classify( X_test ) ) if y == np.argmax( probs ) )
print( float( correct ) / len( y_test ) )

print("Done.") 
