#ifndef __TRAIN_DATA_H__
#define __TRAIN_DATA_H__

#include <vector>
#include "Types.h"

namespace Perceptron {
    
    using namespace std;
    
    class TrainData {
        
    public:
        
        TrainData();
        ~TrainData();
        
        TrainData(const TrainData& rhs);
        TrainData& operator=(const TrainData& rhs);
        
        void swap(TrainData& other);
        
        void addFeatures(vector<FeatureType>& features, bool outputCategory);
        NumberType getDataNumber() const;
        
        inline const vector< vector<FeatureType> > getFeaturesVector() const {
            return vector< vector<FeatureType> >(this->featuresVector);
        }
        
        inline const vector<bool> getOutputCategoriesVector() const {
            return vector<bool>(this->outputCategoriesVector);
        }
        
    private:
        
        NumberType number;
        vector<vector<FeatureType> > featuresVector;
        vector<bool> outputCategoriesVector;
        
        void increaseNumberByOne();
    };
    
}

namespace std {
    template<>
    void swap<Perceptron::TrainData>(Perceptron::TrainData& a, Perceptron::TrainData& b);
}

#endif