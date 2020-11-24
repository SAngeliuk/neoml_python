import numpy as np
import neoml.IsoData as I
import svml

print("Reading iris.svml")

X, y = svml.read( "data/iris.svml" )
weight = np.ones( len(y), dtype=float )

print("Training")

isoData = I.IsoData( init_cluster_count=1, max_cluster_count=20, min_cluster_size=1, max_iteration_count=50,
	min_cluster_distance=0.6, max_cluster_diameter=1, mean_diameter_coef=0.5 )

labels, means, disps = isoData.clusterize( X, weight )

print("Testing")

correct = sum( 1 for y, label in zip( y, labels ) if y == label )
print( float( correct ) / len( y ) )


print("Done.")
