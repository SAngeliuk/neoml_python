/* Copyright © 2017-2020 ABBYY Production LLC

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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// The base class for layers estimating error
// May have 2 to 3 inputs: #0 - the processing result, #1 - the correct result, #2 - the vector weights (optional)
class NEOML_API CLossLayer : public CBaseLayer {
public:
	CLossLayer( IMathEngine& mathEngine, const char* name, bool trainLabels = false);

	void Serialize( CArchive& archive ) override;

	// Total loss weight
	float GetLossWeight() const { return params->GetData().GetValueAt( P_LossWeight ); }
	void SetLossWeight(float _lossWeight) { params->GetData().SetValueAt( P_LossWeight, _lossWeight ); }

	// Turns on and off the labels training mode
	bool TrainLabels() const { return trainLabels; }
	void SetTrainLabels( bool toSet );

	// Retrieves the value of the loss function on the last step
	float GetLastLoss() const { return params->GetData().GetValueAt( P_Loss ); }

	// The maximum loss gradient value
	// The system may not function as intended with very large loss gradient,
	// so we don't recommend changing this value
	float GetMaxGradientValue() const { return params->GetData().GetValueAt( P_MaxGradient ); }
	void SetMaxGradientValue(float maxValue);

	// Tests the layer performance on the basis of the given data and shift vector.
	// The squared L2-difference between the value at the offset point 
	// and the linear approximation based on function gradient
	// is averaged across the BatchSize dimension and returned as the result
	float Test( int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label, int labelSize,
		CConstFloatHandle dataDelta );
	float Test(int batchSize, CConstFloatHandle data, int vectorSize, CConstIntHandle label, int labelSize,
		CConstFloatHandle dataDelta );

	// Tests the layer performance on the basis of data, labels, and dataDelta generated by a uniform random distribution
	// labels and data are of the same size: batchSize * vectorSize
	float TestRandom(CRandom& random, int batchSize, float dataLabelMin, float dataLabelMax, float deltaAbsMax, int vectorSize);
	// Similar to the previous method, but with labels generated as Int [0; labelMax), with size 1
	float TestRandom(CRandom& random, int batchSize, float dataMin, float dataMax, int labelMax, float deltaAbsMax,
		int vectorSize);

protected:
	const CPtr<CDnnBlob>& GetWeights() { return weights; }

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

	// The function that calculates the loss function and its gradient for a vector set
	// The data vectors are stored one after another in the batch. The whole data set is of batchSize * vectorSize size.
	// label is of batchSize * labelSize size, which is generally equal to batchSize * vectorSize
	// result is of batchSize size for the function value and of batchSize * vectorSize size for gradient value
	// You will have to provide the implementation, overloading at least one of these methods, depending on the label types you expect (float or int)
	// IMPORTANT! If lossGradient handle passed to the method is == 0, you don't need to calculate the gradient
	virtual void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient);
	virtual void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle dataLossGradient, CFloatHandle labelLossGradient);
	virtual void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstIntHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient);

private:
	enum TParameter {
		P_LossWeight = 0, // the weight for the loss function
		P_Loss, // the loss value on the last step
		P_LossDivider, // the averaging factor for calculating the loss value
		P_LossGradientDivider, // the averaging factor for calculating the loss gradient (takes lossWeight into account)
		P_MinGradient,
		P_MaxGradient,
		P_Count
	};

	bool trainLabels; // indicates if the first input should be trained against the correct result
	CPtr<CDnnBlob> params; // the blob with all the parameters
	CPtr<CDnnBlob> resultBuffer; // the memory buffer for the errors vector
	CPtr<CDnnBlob> weights;	// the vector weights
	// The blobs that contain loss gradients over the input and the labels
	CObjectArray<CDnnBlob> lossGradientBlobs;

	template<class T>
	float testImpl(int batchSize, CConstFloatHandle data, int vectorSize, CTypedMemoryHandle<const T> label, int labelSize,
		CConstFloatHandle dataDelta);
};

///////////////////////////////////////////////////////////////////////////////////
// CCrossEntropyLossLayer implements a layer that calculates the loss value as cross-entropy between the result and the standard
// By default, softmax function is additionally applied to the results
/*
	May have 2 to 3 inputs:
		#0	The input classification result
			A variable-size blob with the Channels dimension equal to the number of classes
			It contains one of:
			a)	If IsSoftmaxApplied(), the raw classification result (logits).
				Softmax function will be applied to this data.
				This is the default option.

			b)	If !IsSoftmaxApplied(), the classification result as the probability 
				of each class for the given object, values from 0 to 1.
				Softmax will not be applied.

		#1	The correct data labels
			It contains one of:
			a)	A variable-size blob with the Channels dimension equal to the number of classes.
				The values along the Channels dimension specify probabilities for the object to belong to each of the classes.
				If the object belongs to the i class, only the i channel should contain 1, the other channels should be 0.
				Softmax function is never applied to this input.

			b)	A blob with integer data and only one channel. 
				Each value gives the number of the class to which the corresponding object belongs.
			
		#2	[Optional] Object weights
			A variable-size blob with only one channel.
			If you don't connect this input, each object is assumed to have weight 1.
*/

class NEOML_API CCrossEntropyLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CCrossEntropyLossLayer )
public:
	explicit CCrossEntropyLossLayer( IMathEngine& mathEngine );

	// Indicates if softmax function should be applied to input data. True by default.
	// If you turn off the flag, make sure each vector you pass to the input contains only positive numbers making 1 in total.
	void SetApplySoftmax( bool applySoftmax ) { isSoftmaxApplied = applySoftmax; }
	bool IsSoftmaxApplied() const { return isSoftmaxApplied; }

	void Serialize( CArchive& archive ) override;

protected:
	void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient) override;
	void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle dataLossGradient, CFloatHandle labelLossGradient) override;
	void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstIntHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient) override;

private:
	bool isSoftmaxApplied;
};

NEOML_API CLayerWrapper<CCrossEntropyLossLayer> CrossEntropyLoss(
	bool isSoftmaxApplied = true, float lossWeight = 1.0f );

///////////////////////////////////////////////////////////////////////////////////

// CBinaryCrossEntropyLossLayer is a binary variant of cross-entropy
// taking the input of BatchSize * 1 size with the [-1;+1] values
class NEOML_API CBinaryCrossEntropyLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CBinaryCrossEntropyLossLayer )
public:
	explicit CBinaryCrossEntropyLossLayer( IMathEngine& mathEngine );

	// The weight for the positive side of the sigmoid
	// Values over 1 increase recall, values below 1 increase precision
	void SetPositiveWeight( float value );
	float GetPositiveWeight() const;

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	// CLossLayer methods implementation
	void BatchCalculateLossAndGradient( int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient ) override;

private:
	// constants used for calculating the function value
	float positiveWeightMinusOneValue;

	void calculateStableSigmoid( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) const;
};

NEOML_API CLayerWrapper<CBinaryCrossEntropyLossLayer> BinaryCrossEntropyLoss(
	float positiveWeight = 1.0f, float lossWeight = 1.0f );

///////////////////////////////////////////////////////////////////////////////////

// CEuclideanLossLayer implements a layer that calculates the loss function 
// equal to the sum of squared differences between the result and the standard
// The layer has two inputs: #0 - result, #1 - standard
// If |x| > 1, instead of x^2 the 2*(|x| + 1) value will be used (Huber loss)
class NEOML_API CEuclideanLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CEuclideanLossLayer )
public:
	explicit CEuclideanLossLayer( IMathEngine& mathEngine ) : CLossLayer( mathEngine, "CCnnEuclideanLossLayer" ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient) override;
};

NEOML_API CLayerWrapper<CEuclideanLossLayer> EuclideanLoss( float lossWeight = 1.0f );

///////////////////////////////////////////////////////////////////////////////////

// CHingeLossLayer implements a layer that estimates the loss value as max(0, 1 - result * standard)
// The layer has two inputs: #0 - result, #1 - standard
// The standard contains the data in the format: 1 for objects that belong to the class, -1 for the rest
class NEOML_API CHingeLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CHingeLossLayer )
public:
	explicit CHingeLossLayer( IMathEngine& mathEngine ) : CLossLayer( mathEngine, "CCnnHingeLossLayer" ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient) override;
};

NEOML_API CLayerWrapper<CHingeLossLayer> HingeLoss( float lossWeight = 1.0f );

///////////////////////////////////////////////////////////////////////////////////

// CSquaredHingeLossLayer implements a layer that estimates the loss value as max(0, 1 - result * standard)**2
// The layer has two inputs: #0 - result, #1 - standard
// The standard contains the data in the format: 1 for objects that belong to the class, -1 for the rest
// Modified Huber loss is used: loss = -4 * result * standard if result * standard < -1
class NEOML_API CSquaredHingeLossLayer : public CLossLayer {
	NEOML_DNN_LAYER( CSquaredHingeLossLayer )
public:
	explicit CSquaredHingeLossLayer( IMathEngine& mathEngine ) : CLossLayer( mathEngine, "CCnnSquaredHingeLossLayer" ) {}

	void Serialize( CArchive& archive ) override;

protected:
	void BatchCalculateLossAndGradient(int batchSize, CConstFloatHandle data, int vectorSize, CConstFloatHandle label,
		int labelSize, CFloatHandle lossValue, CFloatHandle lossGradient) override;
};

NEOML_API CLayerWrapper<CSquaredHingeLossLayer> SquaredHingeLoss(
	float lossWeight = 1.0f );

} // namespace NeoML
