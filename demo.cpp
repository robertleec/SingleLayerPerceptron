#include <iostream>
#include <vector>
#include "TrainData.h"
#include "SingleLayerPerceptron.h"

using namespace std;

int main(int argc, char *argv[]) {
    
    Perceptron::TrainData trainData;
    
    FeatureType array1[] = {-0.5, -0.5};
    vector<FeatureType> vector1(array1, array1 + 2);
    trainData.addFeatures(vector1, true);
    
    FeatureType array2[] = {-0.5, -0.5};
    vector<FeatureType> vector2(array2, array2 + 2);
    trainData.addFeatures(vector2, true);
    
    FeatureType array3[] = {-0.5, -0.5};
    vector<FeatureType> vector3(array3, array3 + 2);
    trainData.addFeatures(vector3, true);
    
    FeatureType array4[] = {-0.5, -0.5};
    vector<FeatureType> vector4(array4, array4 + 2);
    trainData.addFeatures(vector4, true);
    
    Perceptron::SingleLayerPerceptron singleLayerPerceptron(0.3);
    
    singleLayerPerceptron.learn(trainData);
    
    vector<WeightType> weightVector = singleLayerPerceptron.getWeightVector();
    vector<WeightType>::iterator weightIterator;
    
    cout << "Weights:";
    
    for (weightIterator = weightVector.begin(); weightIterator != weightVector.end(); ++weightIterator) {
        cout << " " << *weightIterator;
    }
    
    cout << endl;
    
    vector<BiasType> biasVector = singleLayerPerceptron.getBiasVector();
    vector<BiasType>::iterator biasIterator;
    
    cout << "Bias:";
    
    for (biasIterator = biasVector.begin(); biasIterator != biasVector.end(); ++biasIterator) {
        cout << " " << *biasIterator;
    }
    
    cout << endl;
    
    return 0;
}