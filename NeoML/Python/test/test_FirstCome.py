import numpy as np
import neoml.FirstCome as F
import svml

print("Reading iris.svml")

X, y = svml.read( "data/iris.svml" )
weight = np.ones( len(y), dtype=float )

print("Training")

first = F.FirstCome( threshold=5 )
labels, means, disps = first.clusterize( X, weight )

print("Testing")

correct = sum( 1 for y, label in zip( y, labels ) if y == label )
print( float( correct ) / len( y ) )


print("Done.")
