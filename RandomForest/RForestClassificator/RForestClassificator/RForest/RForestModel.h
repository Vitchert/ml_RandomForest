#pragma once

#include <vector>
#include <fstream>
#include <numeric>
#include <algorithm>
#include "DecisionTree.h"

struct TRForestModel{
	std::vector<TDecisionTree> forest;	
	std::map<double, double> classTranslation;
	template <typename T>
	double Prediction(const std::vector<T>& features,bool translation = false) {
		int prediction = -1;
		int maxIdx = -1;
		int max = 0;
		std::vector<int> results(classTranslation.size(), 0);

		for (TDecisionTree tree : forest) {		
			prediction = tree.Prediction(features);
			++results[prediction];
			if (results[prediction] > max) {
				max = results[prediction];
				maxIdx = prediction;
			}
		}
		if (translation)
			return classTranslation[maxIdx];
		return maxIdx;
	}
	void SaveToFile(const std::string& modelPath) {
		std::ofstream modelOut(modelPath);
		if (!modelOut.is_open()) {
			std::cout << "Cant create model file\n";
			exit(EXIT_FAILURE);
		}
		modelOut.precision(20);
		modelOut << classTranslation.size() << " ";

		for (auto it = classTranslation.begin(); it != classTranslation.end(); ++it) {
			modelOut << it->first << " " << it->second << " ";
		}
		modelOut << forest.size() << " ";

		for (TDecisionTree tree : forest) {
			tree.SaveToFile(modelOut);
		}
	}
	static TRForestModel LoadFromFile(const std::string& modelPath) {
		std::ifstream modelIn(modelPath);
		if (!modelIn.is_open()) {
			std::cout << "Cant open model file\n";
			exit(EXIT_FAILURE);
		}
		TRForestModel model;

		int classCount;
		double key, val;
		size_t forestSize;
		modelIn >> classCount;
		for (int i = 0; i < classCount; ++i) {
			modelIn >> key >> val;
			model.classTranslation.insert({ key,val });
		}

		modelIn >> forestSize;

		for (int i = 0; i < forestSize; ++i) {
			model.forest.push_back(TDecisionTree::LoadFromFile(modelIn));
		}
		return model;
	};

};