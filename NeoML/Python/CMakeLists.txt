
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake)
include(Version)
include(Settings)
set_global_variables()

project(PythonWrapper)

add_subdirectory(pybind11)
pybind11_add_module(PythonWrapper SHARED
	src/common.cpp
	src/PyArgmaxLayer.cpp
	src/PyAttentionDecoderLayer.cpp
	src/PyBaseConvLayer.cpp
	src/PyChannelwiseConvLayer.cpp
	src/PyConcatChannelsLayer.cpp
	src/PyConvLayer.cpp
	src/PyClustering.cpp
	src/PyDnn.cpp
	src/PyDnnBlob.cpp
	src/PyEltwiseLayer.cpp
	src/PyFullyConnectedLayer.cpp
	src/PyInitializer.cpp
	src/PyLayer.cpp
	src/PyLossLayer.cpp
	src/PyLstmLayer.cpp
	src/PyMathEngine.cpp
	src/PyMemoryFile.cpp
	src/PyMultichannelLookupLayer.cpp
	src/PyPoolingLayer.cpp
	src/PyRandom.cpp
	src/PyReLULayer.cpp
	src/PySinkLayer.cpp
	src/PySolver.cpp
	src/PySourceLayer.cpp
	src/PyTrainingModel.cpp
	src/PyWrapper.cpp
)

find_package(PythonLibs REQUIRED)

target_include_directories(PythonWrapper PRIVATE
    src
    ../include
    ../include/NeoML/FineObjLite
    ${PYTHON_INCLUDE_DIRS}
)

target_link_libraries(PythonWrapper PRIVATE NeoML ${PYTHON_LIBRARIES})
