import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
from scipy.sparse import csr_matrix
import array as ar
from multiprocessing import Process
import time 
from datetime import datetime
 
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_blobs
import svml

from neoml.KMeans import KMeans as KMeansNeo
from sklearn.cluster import KMeans as KMeansSk
from neoml.Hierarchical import Hierarchical as HierarchicalNeo
from sklearn.cluster import AgglomerativeClustering as HierarchicalSk

from neoml.Linear import LinearClassifier as LinearBinaryNeo
from sklearn.linear_model import LogisticRegression as LinearBinarySk
from neoml.SVM import SvmClassifier as SvmNeo
from sklearn.svm import SVC as SvmSk
from neoml.OneVersusAll import OneVersusAllClassifier as OneVsAllNeo
from sklearn.multiclass import OneVsRestClassifier as OneVsAllSk
from neoml.DecisionTree import DecisionTreeClassifier as DecisionTreeClNeo
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClSk
from neoml.GradientBoost import GradientBoostClassifier as GradientBoostClNeo
from sklearn.ensemble import GradientBoostingClassifier as GradientBoostClSk

def neoTrainKMeans( X, y, w ):
	return KMeansNeo( max_iteration_count=50, init_cluster_count=3 ).clusterize( X, w )

def skTrainKMeans( X, y, w, centers_centers ):
	return KMeansSk( max_iter=50, n_clusters=3, algorithm="full", n_init=1, random_state=0, init=centers_centers ).fit( X, sample_weight=w )

def neoTrainHierarchical( X, y, w ):
	return HierarchicalNeo( min_cluster_count=3, max_cluster_distance=100 ).clusterize( X, w )

def skTrainHierarchical( X, y, w ):
	return HierarchicalSk( linkage='average', n_clusters=None, distance_threshold=100 ).fit( X )

def neoTrainLinearBinaryClassifier( X, y, w ):
	return LinearBinaryNeo( loss='binomial', l1_reg=0, max_iteration_count=100 ).train( X, y, w )

def skTrainLinearBinaryClassifier( X, y, w ):
	return LinearBinarySk( random_state=0, penalty='l2', solver='liblinear', max_iter=100 ).fit( X, y, w )

def neoTrainLinearSVM( X, y, w ):
	return SvmNeo( kernel='linear', max_iteration_count=1000 ).train( X, y, w )

def skTrainLinearSVM( X, y, w ):
	return SvmSk( kernel='linear', max_iter=1000 ).fit( X, y, w )

def neoTrainRBFSVM( X, y, w ):
	return SvmNeo( kernel='rbf', max_iteration_count=1000 ).train( X, y, w )

def skTrainRBFSVM( X, y, w ):
	return SvmSk( kernel='rbf', max_iter=1000 ).fit( X, y, w )

def neoTrainOneVsAllLinearSVM( X, y, w ):
	return OneVsAllNeo( SvmNeo( kernel='linear', max_iteration_count=1000 ) ).train( X, y, w )

def skTrainOneVsAllLinearSVM( X, y, w ):
	return OneVsAllSk( SvmSk( kernel='linear', max_iter=1000 ) ).fit( X, y )

def neoTrainGradientBoostingClassifier( X, y, w ):
	return GradientBoostClNeo( loss ='binomial', iteration_count=10 ).train( X, y, w )

def skTrainGradientBoostingClassifier( X, y, w ):
	return GradientBoostClSk( loss='deviance', n_estimators=10, random_state=0 ).fit( X, y, w )

def neoTrainDecisionTreeClassifier( X, y, w ):
	return DecisionTreeClNeo( criterion='gini', min_split_size=2, min_subset_part=0.10, min_subset_size=128, max_tree_depth=10 ).train( X, y, w )

def skTrainDecisionTreeClassifier( X, y, w ):
	return DecisionTreeClSk( criterion='gini', min_samples_split=2, min_weight_fraction_leaf=0.10, min_samples_leaf=128, max_depth=10 ).fit( X, y, w )

###################################################################################################################
def neoScoreClassifier( classifierModel, X_test, y_test ):
	correct = sum( 1 for y, probs in zip( y_test, classifierModel.classify( X_test ) ) if y == np.argmax( probs ) )
	print( f"score: {float( correct ) / len( y_test )}" )

def neoScoreClusterizer( clusterizationRes, X_test, y_test ):
	correct = sum( 1 for y, label in zip( y_test, clusterizationRes[0] ) if y == label )
	print( f"score: {float( correct ) / len( y_test )}" )

def skScoreClassifier( classifier, X_test, y_test ):
	score = classifier.score( X_test, y_test )
	print( f"score: {score}" )

def skScoreClusterizer( clusterizer, X_test, y_test ):
	correct = sum( 1 for y, label in zip( y_test, clusterizer.labels_ ) if y == label )
	print( f"score: {float( correct ) / len( y_test )}" )

ClassificationAlgorithms = {
	"Linear binary NeoML": [neoTrainLinearBinaryClassifier, neoScoreClassifier],
	"Linear binary sklearn": [skTrainLinearBinaryClassifier, skScoreClassifier],
	"Linear SVM NeoML": [neoTrainLinearSVM, neoScoreClassifier],
	"Linear SVM sklearn": [skTrainLinearSVM, skScoreClassifier],
	"RBF SVM NeoML": [neoTrainRBFSVM, neoScoreClassifier],
	"RBF SVM sklearn": [skTrainRBFSVM, skScoreClassifier],
	"OneVsAll with Linear SVM NeoML": [neoTrainOneVsAllLinearSVM, neoScoreClassifier],
	"OneVsAll with Linear SVM sklearn": [skTrainOneVsAllLinearSVM, skScoreClassifier],
	"Gradient boosting classifier NeoML": [neoTrainGradientBoostingClassifier, neoScoreClassifier],
	"Gradient boosting classifier sklearn": [skTrainGradientBoostingClassifier, skScoreClassifier],
	"Decision tree classifier NeoML": [neoTrainDecisionTreeClassifier, neoScoreClassifier],
	"Decision tree classifier sklearn": [skTrainDecisionTreeClassifier, skScoreClassifier]
}

ClusteringAlgorithms = {
	"KMeans NeoML": [neoTrainKMeans, neoScoreClusterizer],
	"KMeans sklearn": [skTrainKMeans, skScoreClusterizer],
	"Hierarchical NeoML": [neoTrainHierarchical, neoScoreClusterizer],
	"Hierarchical sklearn": [skTrainHierarchical, skScoreClusterizer]
}

def timeoutFunc( timeout, func ):
#	p = Process( target=func )
#	p.start()
#	p.join( timeout )
#	if p.is_alive():
#		print( "end of time" )
#		p.terminate()
#		p.join()
#		return False
	func()
	return True

def measure( func, n_runs, timeout=None ):
	tic = time.perf_counter()
	for i in range(0, n_runs):
		if timeoutFunc( timeout, func ) == False:
			return
	print("time: {0}".format( (time.perf_counter() - tic) / n_runs ))

TestFunctions = {
	"measure_train_time": lambda trainFunc, scoreFunc, n_runs=1, timeout=None: measure( trainFunc, n_runs, timeout ),
	"calculate_score": lambda trainFunc, scoreFunc, n_runs=1, timeout=None: timeoutFunc( timeout, scoreFunc )
}

def runTests( X_train, X_test, y_train, y_test, w_train=None, w_test=None, n_runs=1, timeout=None, algorithms=ClassificationAlgorithms, testFuncs=TestFunctions ):
	if w_train is None:
		w_train = np.ones( len(y_train), dtype=float )
	if w_test is None:
		w_test = np.ones( len(y_test), dtype=float )

	for desc, algo in algorithms.items():
		print( "\n{0} {1}".format(datetime.now().time(), desc), flush=True )
		if desc == "KMeans sklearn":
			_, centers, _ = neoTrainKMeans( X, y, w )
			trainFunc = lambda: algo[0]( X_train, y_train, w_train, centers )
		else:
			trainFunc = lambda: algo[0]( X_train, y_train, w_train )

		scoreFunc = lambda: algo[1]( trainFunc(), X_test, y_test )
		for testFuncDesc, testFunc in testFuncs.items():
			print( testFuncDesc, flush=True )
			testFunc( trainFunc, scoreFunc, n_runs, timeout )

def splitDataAndRunTests( X, y, w=None, n_runs=1, timeout=None, algorithms=ClassificationAlgorithms, testFuncs=TestFunctions ):
	if w is None:
		w = np.ones( len(y), dtype=float )
	X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=.2)
	X_train_sparse = csr_matrix( X_train, dtype=np.float32 ) 
	X_test_sparse = csr_matrix( X_test, dtype=np.float32 ) 
	runTests( X_train, X_test, y_train, y_test, w_train, w_test, n_runs=n_runs, timeout=timeout, algorithms=algorithms, testFuncs=testFuncs )

def createClassificationDatasetAndTest( n_samples=5000, n_features=20, n_classes=2, n_informative=2, n_runs=1000, timeout=2, algorithms=ClassificationAlgorithms, testFuncs=TestFunctions ):
	print(f"\nTesting on the dataset = make_classification( n_samples={n_samples}, n_features={n_features}, n_classes={n_classes} ), {n_runs} runs with timeout {timeout}s", flush=True)
	X, y = make_classification( n_samples=n_samples, n_features=n_features, n_classes=n_classes, n_informative=n_informative )
	splitDataAndRunTests( X, y, n_runs=n_runs, timeout=timeout, algorithms=algorithms, testFuncs=testFuncs )

def createClusteringDatasetAndTest( n_samples=5000, n_features=20, centers=3, n_runs=1000, timeout=2, algorithms=ClusteringAlgorithms, testFuncs=TestFunctions ):
	print(f"\nTesting on the dataset = make_blobs( n_samples={n_samples}, n_features={n_features}, centers={centers} ), {n_runs} runs with timeout {timeout}s", flush=True)
	X, y = make_blobs( n_samples=n_samples, n_features=n_features, centers=centers )
	splitDataAndRunTests( X, y, n_runs=n_runs, timeout=timeout, algorithms=algorithms, testFuncs=testFuncs )

def testOnInternalClassificationDataset( dataset, n_runs=10, timeout=500, algorithms=ClassificationAlgorithms, testFuncs=TestFunctions ):
	print( f"Testing on internal dataset {dataset} with timeout={timeout}, n_runs={n_runs}", flush=True )
	X, y = svml.read( dataset + ".train.svml" )
	X_test, y_test = svml.read( dataset + ".test.svml", min_feature_count=X.shape[1] )
	runTests( X, X_test, y, y_test, n_runs=n_runs, timeout=timeout, algorithms=algorithms, testFuncs=testFuncs )

myAlgos = ["Linear SVM NeoML", "Linear SVM sklearn"]
algosToTestBinary = ["Linear binary NeoML", "Linear binary sklearn", "Linear SVM NeoML", "Linear SVM sklearn", "RBF SVM NeoML", "RBF SVM sklearn", "Decision tree classifier NeoML", "Decision tree classifier sklearn"]
algosToTestMulti = ["OneVsAll with Linear SVM NeoML", "OneVsAll with Linear SVM sklearn", "Gradient boosting classifier NeoML", "Gradient boosting classifier sklearn"]

algorithmsMy={ key: ClassificationAlgorithms[key] for key in myAlgos }
algorithmsBinary={ key: ClassificationAlgorithms[key] for key in algosToTestBinary }
algorithmsMulti={ key: ClassificationAlgorithms[key] for key in algosToTestMulti }
testFuncNames = [ "measure_train_time", "calculate_score" ]
testFuncs={ key: TestFunctions[key] for key in testFuncNames }

#testOnInternalClassificationDataset( "data/news20bin", n_runs=100, timeout=2, algorithms=algorithmsBinary )
#testOnInternalClassificationDataset( "data/news20", algorithms=algorithmsMulti )

#print("Binary classification (small, medium, big):")
createClassificationDatasetAndTest( algorithms=algorithmsMy, n_runs=1 )
#createClassificationDatasetAndTest( n_samples=20000, n_runs=100, timeout=50 )
#createClassificationDatasetAndTest( n_samples=100000, n_runs=5, timeout=500 )
#
#print("\n\nMulti-class classification (small, medium, big):")
#createClassificationDatasetAndTest( n_classes=10, n_informative=5 )
#createClassificationDatasetAndTest( n_samples=20000, n_classes=10, n_informative=5, n_runs=100, timeout=50 )
#createClassificationDatasetAndTest( n_samples=100000, n_classes=10, n_informative=5, n_runs=5, timeout=500 )
#
#print("Clustering (small, medium, big):")
#createClusteringDatasetAndTest()
#createClusteringDatasetAndTest( n_samples=20000, n_runs=100, timeout=50 )
#createClusteringDatasetAndTest( n_samples=100000, n_runs=5, timeout=500 )

#createClassificationDatasetAndTest( n_runs=5, timeout=20, algorithms=algorithms, testFuncs=testFuncs )
#createClassificationDatasetAndTest( n_runs=1, timeout=100, n_samples=20000, algorithms=algorithms, testFuncs=testFuncs )
#createClassificationDatasetAndTest( n_runs=5, timeout=20, n_classes=10, n_informative=5, algorithms=algorithms, testFuncs=testFuncs )

#algosToTest = ["Hierarchical NeoML", "Hierarchical sklearn"]
#testFuncs = { "measure_train_time", "calculate_score" }
#createClusteringDatasetAndTest( n_samples=2000, n_runs=10, testFuncs={ key: TestFunctions[key] for key in testFuncs } )

print("Done.") 
