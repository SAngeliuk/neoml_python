import numpy as np
from scipy.sparse import csr_matrix
import neoml.SVM as S
import neoml.OneVersusAll as O
import svml
import time

print("Reading news20.train.svml")

X, y = svml.read( "data/news20.train.svml" )
weight = np.ones( len(y), dtype=float )

print("Training")

tic = time.perf_counter()
svm = S.SvmClassifier( kernel='linear' )
one = O.OneVersusAllClassifier( svm )

model = one.train( X, y, weight )
print(f"time elapsed {time.perf_counter()-tic}")

print("Reading news20.test.svml")

X_test, y_test = svml.read( "data/news20.test.svml" )

print("Testing")

correct = sum( 1 for y, probs in zip( y_test, model.classify( X_test ) ) if y == np.argmax( probs ) )
print( float( correct ) / len( y_test ) )

print("Done.") 
