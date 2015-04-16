#ifndef __SINGLE_LAYER_PERCEPTRON_H__
#define __SINGLE_LAYER_PERCEPTRON_H__

#include <vector>
#include "Types.h"

namespace Perceptron {
    
    using namespace std;
    
    class TrainData;
    
    class SingleLayerPerceptron {
        
    public:
        
        SingleLayerPerceptron();
        SingleLayerPerceptron(LearningRateType learningRate);
        ~SingleLayerPerceptron();
        
        SingleLayerPerceptron(const SingleLayerPerceptron& rhs);
        SingleLayerPerceptron& operator=(const SingleLayerPerceptron& rhs);
        
        void swap(SingleLayerPerceptron& other);
        
        void learn(const TrainData& trainData);
        bool getClassifier(const vector<FeatureType>& features);
        
        inline const vector<WeightType> getWeightVector() {
            return vector<WeightType>(this->weightVector);
        }
        
        BiasType getBias();
        
    private:
        
        LearningRateType learningRate;
        
        vector<WeightType> weightVector;
        BiasType bias;
        
        void updateParameters(const vector<FeatureType>& featureVector, bool outputCategory);
        void resetWeightVector(size_t size);
    };

}

namespace std {
    template<>
    void swap<Perceptron::SingleLayerPerceptron>(Perceptron::SingleLayerPerceptron& a, Perceptron::SingleLayerPerceptron& b);
}

#endif