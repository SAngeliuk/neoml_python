/* Copyright © 2017-2021 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#include <common.h>
#pragma hdrstop

#include "PyEltwiseLayer.h"

class CPyEltwiseSumLayer : public CPyLayer {
public:
	explicit CPyEltwiseSumLayer( CEltwiseSumLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPyEltwiseMulLayer : public CPyLayer {
public:
	explicit CPyEltwiseMulLayer( CEltwiseMulLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPyEltwiseNegMulLayer : public CPyLayer {
public:
	explicit CPyEltwiseNegMulLayer( CEltwiseNegMulLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}
};

//------------------------------------------------------------------------------------------------------------

class CPyEltwiseMaxLayer : public CPyLayer {
public:
	explicit CPyEltwiseMaxLayer( CEltwiseMaxLayer& layer, CPyMathEngineOwner& mathEngineOwner ) : CPyLayer( layer, mathEngineOwner ) {}
};

void InitializeEltwiseLayer( py::module& m )
{
	py::class_<CPyEltwiseSumLayer, CPyLayer>(m, "EltwiseSum")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseSumLayer( *layer.Layer<CEltwiseSumLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseSumLayer> eltwise = new CEltwiseSumLayer( mathEngine );
			eltwise->SetName( name == "" ? findFreeLayerName( dnn, "EltwiseSumLayer" ).c_str() : name.c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseSumLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyEltwiseMulLayer, CPyLayer>(m, "EltwiseMul")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseMulLayer( *layer.Layer<CEltwiseMulLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseMulLayer> eltwise = new CEltwiseMulLayer( mathEngine );
			eltwise->SetName( name == "" ? findFreeLayerName( dnn, "EltwiseMulLayer" ).c_str() : name.c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseMulLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyEltwiseNegMulLayer, CPyLayer>(m, "EltwiseNegMul")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseNegMulLayer( *layer.Layer<CEltwiseNegMulLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseNegMulLayer> eltwise = new CEltwiseNegMulLayer( mathEngine );
			eltwise->SetName( name == "" ? findFreeLayerName( dnn, "EltwiseNegMulLayer" ).c_str() : name.c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseNegMulLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;

//------------------------------------------------------------------------------------------------------------

	py::class_<CPyEltwiseMaxLayer, CPyLayer>(m, "EltwiseMax")
		.def( py::init([]( const CPyLayer& layer )
		{
			return CPyEltwiseMaxLayer( *layer.Layer<CEltwiseMaxLayer>(), layer.MathEngineOwner() );
		}))
		.def( py::init([]( const std::string& name, const py::list& layers, const py::list& outputs )
		{
			CDnn& dnn = layers[0].cast<CPyLayer>().Dnn();
			IMathEngine& mathEngine = dnn.GetMathEngine();

			CPtr<CEltwiseMaxLayer> eltwise = new CEltwiseMaxLayer( mathEngine );
			eltwise->SetName( name == "" ? findFreeLayerName( dnn, "EltwiseMaxLayer" ).c_str() : name.c_str() );
			dnn.AddLayer( *eltwise );

			for( int i = 0; i < layers.size(); i++ ) {
				eltwise->Connect( i, layers[i].cast<CPyLayer>().BaseLayer(), outputs[i].cast<int>() );
			}

			return new CPyEltwiseMaxLayer( *eltwise, layers[0].cast<CPyLayer>().MathEngineOwner() );
		}) )
	;
}
