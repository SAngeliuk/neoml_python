import numpy as np
import neoml
import time

def read_list( file_path ) :
	file = open(file_path, "r")

	data = []

	for line in file:
		cur = line.split(",")

		y = []
		y.append( int(cur[0]) )

		for i in range(1, len(cur) ):
			y.append( int(cur[i]) )

		data.append( y )

	return data

def build_dnn( math_engine, input_class_count, output_class_count, output_seq_len, hidden_size, embedding_size ) :
	dnn = neoml.Dnn.Dnn(math_engine)

	source = neoml.Dnn.Source(dnn, "data")
	init_data = neoml.Dnn.Source(dnn, "initData")
	label = neoml.Dnn.Source(dnn, "label")

	direct_encoder_input = source

	if embedding_size > 0 :
		lookup = neoml.Dnn.MultichannelLookup( source, [(input_class_count, embedding_size)], "lookup" ); 	

		lookup.initialize( neoml.Initializer.Uniform() )

		direct_encoder_input = lookup

	direct_encoder = neoml.Dnn.Lstm( direct_encoder_input, hidden_size = hidden_size, recurrent_activation="hard_sigmoid", name="directEncoder" )

	reverse_encoder = neoml.Dnn.Lstm( direct_encoder_input, hidden_size = hidden_size, recurrent_activation="hard_sigmoid", reverse_seq=True, name="reversedEncoder" )

	encoder_output = neoml.Dnn.ConcatChannels( ( direct_encoder, reverse_encoder ), "encoderOutput" )

	decoder = neoml.Dnn.AttentionDecoder( ( encoder_output, init_data ), score="additive", hidden_size=hidden_size, output_object_size=output_class_count, output_seq_len=output_seq_len, name="decoder" )

	argmax = neoml.Dnn.Argmax( decoder )

	output = neoml.Dnn.Sink( argmax, "output" )

	loss = neoml.Dnn.CrossEntropyLoss(( decoder, label ), softmax=False, name="loss")

	return ( dnn, decoder, loss )

def run_and_learn_once( dnn, decoder, loss, word_blob, label_blob, prev_label_blob ) :
	decoder.set_output_seq_len( label_blob.batch_len() )
	dnn.learn( { "data" : word_blob, "label" : label_blob, "initData" : prev_label_blob } )
	return loss.last_loss()

def run_once( dnn, decoder, loss, word_blob, label_blob, prev_label_blob ) :
	decoder.set_output_seq_len( label_blob.batch_len() )
	dnn.run( { "data" : word_blob, "label" : label_blob, "initData" : prev_label_blob } )
	return loss.last_loss()


def createIntBatch( math_engine, process_type, data, index, batch_size ) :
	batch_len = 0
	for i in range(0, batch_size):
		batch_len = max( batch_len, len( data[index + i] ) )

	if process_type != "SPT_Whole":
		batch_len -= 1

	buffer = np.zeros( batch_size * batch_len, dtype=np.int32 )

	for i in range( 0, batch_size ):
		first = 0
		last = len( data[index + i] )
		if process_type == "SPT_SkipFirst" :
			first += 1
		if process_type == "SPT_SkipLast" :
			last -= 1
		for j in range( first, last ):
			buffer[(j - first) * batch_size + i] = data[index + i][j]

	return neoml.Blob.data_blob( math_engine, batch_len, batch_size, 1, "int32", buffer )


def createFloatBatch( math_engine, process_type, data, index, batch_size, class_count ) :
	batch_len = 0
	for i in range(0, batch_size):
		batch_len = max( batch_len, len( data[index + i] ) )

	if process_type != "SPT_Whole":
		batch_len -= 1

	buffer = np.zeros( batch_size * batch_len * class_count, dtype=np.int32 )

	for i in range( 0, batch_size ):
		first = 0
		last = len( data[index + i] )
		if process_type == "SPT_SkipFirst" :
			first += 1
		if process_type == "SPT_SkipLast" :
			last -= 1
		for j in range( first, last ):
			buffer[(j - first) * batch_size * class_count + i * class_count + data[index + i][j]] = 1.0

	return neoml.Blob.data_blob( math_engine, batch_len, batch_size, class_count, "float32", buffer )

#-------------------------------------------------------------------------------------------------------------

OutputClassCount = 14
BatchSize = 500

#-------------------------------------------------------------------------------------------------------------


train_words = read_list("data/dates.train.data")
train_labels = read_list("data/dates.train.label")

test_words = read_list("data/dates.test.data")
test_labels = read_list("data/dates.test.label")

init_label = [[1]] * BatchSize

#-------------------------------------------------------------------------------------------------------------

math_engine = neoml.MathEngine.GpuMathEngine( 0 )

dnn, decoder, loss = build_dnn( math_engine, input_class_count=59, output_class_count=OutputClassCount, output_seq_len=1, hidden_size=256, embedding_size=32 )

solver = neoml.Solver.NesterovGradient(math_engine, learning_rate=0.001, l1=0.0, l2=0.0, moment_decay_rate=0.9, max_gradient_norm=5.0, second_moment_decay_rate=0.999, epsilon=1e-8)

dnn.solver = solver

for epoch in range(0, 20):
	start_time = time.perf_counter()
	epoch_loss = 0.0
	index = 0
	while index < len(train_words):
		word_data = createIntBatch( math_engine, "SPT_Whole", train_words, index, min( BatchSize, len( train_words ) - index ) )
		label_data = createIntBatch( math_engine, "SPT_SkipFirst", train_labels, index, min( BatchSize, len( train_words ) - index ) )
		prev_label_data = createFloatBatch( math_engine, "SPT_SkipLast", train_labels, index, min( BatchSize, len( train_words ) - index ), OutputClassCount )

		epoch_loss += run_and_learn_once( dnn, decoder, loss, word_data, label_data, prev_label_data )
		index += BatchSize

	epoch_loss /= len(train_words) / BatchSize

	epoch_test_loss = 0

	index = 0
	while index < len(test_words):
		word_data = createIntBatch( math_engine, "SPT_Whole", test_words, index, min(BatchSize, len(test_words) - index) )
		label_data = createIntBatch( math_engine, "SPT_SkipFirst", test_labels, index, min(BatchSize, len(test_words) - index) )
		init_data = createFloatBatch( math_engine, "SPT_Whole", init_label, 0, min(BatchSize, len(test_words) - index), OutputClassCount )
	
		epoch_test_loss += run_once( dnn, decoder, loss, word_data, label_data, init_data )
		index += BatchSize
		
	epoch_test_loss /= len(test_words) / BatchSize

	print( "Epoch #" + str(epoch) )
	print( "Epoch Time: " + str(time.perf_counter() - start_time) )
	print( "Epoch avg loss: " + str(epoch_loss) )
	print( "Epoch avg test loss: " + str(epoch_test_loss) )

print( "The end" )
	

 