import numpy as np
import neoml.Hierarchical as H
import svml

print("Reading iris.svml")

X, y = svml.read( "data/iris.svml" )
weight = np.ones( len(y), dtype=float )

print("Training")

hierarchical = H.Hierarchical( min_cluster_count=3, max_cluster_distance=5 )
labels, means, disps = hierarchical.clusterize( X, weight )

print("Testing")

correct = sum( 1 for y, label in zip( y, labels ) if y == label )
print( float( correct ) / len( y ) )


print("Done.")
