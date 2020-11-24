import numpy as np
import neoml
import svml

print("Reading iris.svml")

X, y = svml.read("data/iris.svml")
weight = np.ones(len(y), dtype=float)

print("Training")

kmeans = neoml.KMeans.KMeans(max_iteration_count=50, init_cluster_count=3)
labels, means, disps = kmeans.clusterize(X, weight)

print("Testing")

correct = sum(1 for y, label in zip(y, labels) if y == label)
print(float(correct) / len(y))

print("Done.")
