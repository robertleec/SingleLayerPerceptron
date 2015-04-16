#include "SingleLayerPerceptron.h"
#include "TrainData.h"

#define DEFAULT_LEARNING_RATE 0.2

namespace Perceptron {
    
    using namespace std;
    
    typedef double CategoryValueType;
    
    SingleLayerPerceptron::SingleLayerPerceptron() {
        this->learningRate = DEFAULT_LEARNING_RATE;
    }
    
    SingleLayerPerceptron::SingleLayerPerceptron(LearningRateType learningRate) {
        this->learningRate = learningRate;
    }
    
    SingleLayerPerceptron::~SingleLayerPerceptron() {
        
    }
    
    SingleLayerPerceptron::SingleLayerPerceptron(const SingleLayerPerceptron& rhs):learningRate(rhs.learningRate), weightVector(rhs.weightVector), bias(rhs.bias) {
        
    }
    
    SingleLayerPerceptron& SingleLayerPerceptron::operator=(const SingleLayerPerceptron& rhs) {
        SingleLayerPerceptron temp(rhs);
        this->swap(temp);
        return *this;
    }
    
    void SingleLayerPerceptron::swap(SingleLayerPerceptron& other) {
        using std::swap;
        swap(this->learningRate, other.learningRate);
        swap(this->weightVector, other.weightVector);
        swap(this->bias, other.bias);
    }
    
    void SingleLayerPerceptron::learn(const TrainData &trainData) {
        
        bool isLearingFinished = false;
        
        vector< vector<FeatureType> > featuresVector = trainData.getFeaturesVector();
        vector<bool> outputCategoriesVector = trainData.getOutputCategoriesVector();
        
        vector< vector<FeatureType> >::iterator featuresIterator;
        vector<bool>::iterator outputCategoryIterator;
        
        while (isLearingFinished == false) {
            
            isLearingFinished = true;
            
            for (featuresIterator = featuresVector.begin(), outputCategoryIterator = outputCategoriesVector.begin(); featuresIterator != featuresVector.end() && outputCategoryIterator != outputCategoriesVector.end(); ++featuresIterator, ++outputCategoryIterator) {
                
                if (getClassifier(*featuresIterator) != *outputCategoryIterator) {
                    this->updateParameters(*featuresIterator, *outputCategoryIterator);
                    
                    isLearingFinished = false;
                }
            }
        }
    }
    
    void SingleLayerPerceptron::updateParameters(const vector<FeatureType>& featureVector, bool outputCategory) {
        
        if ((this->weightVector).size() != featureVector.size()) {
            this->resetWeightVector(featureVector.size());
        }
        
        vector<FeatureType>::const_iterator featureIterator;
        vector<WeightType>::iterator weightIterator;
        
        for (featureIterator = featureVector.begin(), weightIterator = (this->weightVector).begin(); featureIterator != featureVector.end() && weightIterator != (this->weightVector).end(); ++featureIterator, ++weightIterator) {
            
            if (outputCategory == true) {
                *weightIterator += (*featureIterator) * (1) * (this->learningRate);
                this->bias += (1) * (this->learningRate);
            } else {
                *weightIterator += (*featureIterator) * (-1) * (this->learningRate);
                this->bias += (-1) * (this->learningRate);
            }
        }
    }
    
    void SingleLayerPerceptron::resetWeightVector(size_t size) {
        this->weightVector = vector<WeightType>(size);
    }
    
    bool SingleLayerPerceptron::getClassifier(const vector<FeatureType>& features) {
        CategoryValueType categoryValue = 0;
        
        vector<FeatureType>::const_iterator featureIterator;
        vector<WeightType>::iterator weightIterator;
        
        for (featureIterator = features.begin(), weightIterator = (this->weightVector).begin(); featureIterator != features.end() && weightIterator != (this->weightVector).end(); ++featureIterator, ++weightIterator) {
            
            categoryValue += (*featureIterator) * (*weightIterator);
        }
        
        categoryValue += this->bias;
        
        if (categoryValue > 0) {
            return true;
        }
        
        return false;
    }
    
    BiasType SingleLayerPerceptron::getBias() {
        return this->bias;
    }
}

namespace std {
    template<>
    void swap<Perceptron::SingleLayerPerceptron>(Perceptron::SingleLayerPerceptron& a, Perceptron::SingleLayerPerceptron& b) {
        a.swap(b);
    }
}