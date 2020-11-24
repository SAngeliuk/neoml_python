import numpy as np
import neoml.DecisionTree as D
import svml

print("Reading news20bin.train.svml")

X, y = svml.read( "data/news20bin.train.svml" )
weight = np.ones( len(y), dtype=float )

print("Training")

decision = D.DecisionTreeClassifier( criterion='information_gain', min_subset_part=0.10, min_subset_size=128, max_tree_depth=10 )
model = decision.train( X, y, weight )

print("Reading news20bin.test.svml")

X_test, y_test = svml.read( "data/news20bin.test.svml" )

print("Testing")

correct = sum( 1 for y, probs in zip( y_test, model.classify( X_test ) ) if y == np.argmax(probs) )
print( float( correct ) / len( y_test ) )

print("Done.") 
