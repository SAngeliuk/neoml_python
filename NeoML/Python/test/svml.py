import numpy as np
from scipy.sparse import csr_matrix

def read( file_path, min_feature_count=0 ) :
	file = open(file_path, "r")

	data = []
	column = []
	row = [0]
	y = []
	rowCount = 0;
	columnCount = 0
	elementCount = 0

	for line in file:
		cur = line.split(" ")

		y.append( int(cur[0]) )

		for i in range(1, len(cur) ):
			item = cur[i].split(":")
			data.append( float(item[1]) )
			column.append( int(item[0]) )
			if int(item[0]) + 1 > columnCount:
				columnCount = int(item[0]) + 1
			elementCount += 1

		rowCount += 1
		row.append( elementCount )

	X = csr_matrix( ( np.array( data, np.float32 ), np.array( column, np.int32 ), row ), shape=( rowCount, max(columnCount, min_feature_count) ) )

	return ( X, y )

def correct( file_path ) :
	file = open(file_path, "r")
	fileW = open("res.txt", "w")

	data = []
	column = []
	row = [0]
	y = []
	rowCount = 0;
	columnCount = 0
	elementCount = 0

	for line in file:
		cur = line.split(" ")

		y.append( int(cur[0]) )

		fileW.write( str( int( cur[0] ) - 1 ) )

		for i in range(1, len(cur) ):
			item = cur[i].split(":")
			data.append( float(item[1]) )
			column.append( int(item[0]) )
			if int(item[0]) + 1 > columnCount:
				columnCount = int(item[0]) + 1
			elementCount += 1
			s = str(int(item[0]) - 1 ) + ":" + item[1]
			fileW.write( " " + s )

		rowCount += 1
		row.append( elementCount )

	X = csr_matrix( ( np.array( data, np.float32 ), np.array( column, np.int32 ), row ), shape=( rowCount, columnCount ) )

	return ( X, y )

