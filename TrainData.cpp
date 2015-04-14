#include "TrainData.h"
#include "assert.h"

namespace Perceptron {
    
    using namespace std;
    
    TrainData::TrainData() {
        this->number = 0;
    }
    
    TrainData::~TrainData() {
        
    }
    
    TrainData::TrainData(const TrainData& rhs):number(rhs.number), featuresVector(rhs.featuresVector), outputCategoriesVector(rhs.outputCategoriesVector) {
        
    }
    
    TrainData& TrainData::operator=(const TrainData& rhs) {
        TrainData temp(rhs);
        this->swap(temp);
        return *this;
    }
    
    void TrainData::swap(TrainData& other) {
        using std::swap;
        swap(this->number, other.number);
        swap(this->featuresVector, other.featuresVector);
        swap(this->outputCategoriesVector, other.outputCategoriesVector);
    }
    
    void TrainData::addFeatures(vector<FeatureType>& features, bool outputCategory) {
        (this->featuresVector).push_back(features);
        (this->outputCategoriesVector).push_back(outputCategory);
        increaseNumberByOne();
    }
    
    NumberType TrainData::getDataNumber() const {
        return this->number;
    }
    
    void TrainData::increaseNumberByOne() {
        ++(this->number);
    }
}

namespace std {
    template<>
    void swap<Perceptron::TrainData>(Perceptron::TrainData& a, Perceptron::TrainData& b) {
        a.swap(b);
    }
}
