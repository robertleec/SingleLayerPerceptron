#include <iostream>
#include <vector>
#include "TrainData.h"
#include "SingleLayerPerceptron.h"

using namespace std;

int main(int argc, char *argv[]) {
    
    Perceptron::TrainData trainData;
    
    FeatureType array1[] = {-0.5, -0.5};
    vector<FeatureType> vector1(array1, array1 + 2);
    trainData.addFeatures(vector1, CATEGORY_TYPE_POSITIVE);
    
    FeatureType array2[] = {-0.5, 0.5};
    vector<FeatureType> vector2(array2, array2 + 2);
    trainData.addFeatures(vector2, CATEGORY_TYPE_POSITIVE);
    
    FeatureType array3[] = {0.5, -0.5};
    vector<FeatureType> vector3(array3, array3 + 2);
    trainData.addFeatures(vector3, CATEGORY_TYPE_POSITIVE);
    
    FeatureType array4[] = {0.5, 0.5};
    vector<FeatureType> vector4(array4, array4 + 2);
    trainData.addFeatures(vector4, CATEGORY_TYPE_NEGATIVE);
    
    Perceptron::SingleLayerPerceptron singleLayerPerceptron(0.3);
    
    singleLayerPerceptron.learn(trainData);
    
    vector<WeightType> weightVector = singleLayerPerceptron.getWeightVector();
    vector<WeightType>::iterator weightIterator;
    
    cout << "Weights:";
    
    for (weightIterator = weightVector.begin(); weightIterator != weightVector.end(); ++weightIterator) {
        cout << " " << *weightIterator;
    }
    
    cout << endl;
    
    cout << "Bias:" << " " << singleLayerPerceptron.getBias() << endl;
    
    return 0;
}