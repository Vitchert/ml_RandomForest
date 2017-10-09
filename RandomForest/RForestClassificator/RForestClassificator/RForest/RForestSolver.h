#pragma once
#include <iostream>
#include "RForestModel.h"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono> 
#include "windows.h"

class TRForestSolver {
private:
	TDataset dataset;
public:
	void AddDataset(const TDataset& data) {
		dataset = data;
	}
	void Add(const std::vector<double>& features, const double goal, const double weight = 1.) {
		dataset.featuresMatrix.push_back(features);
		dataset.goals.push_back(goal);
		dataset.weights.push_back(weight);
	}

	std::mutex g_lock;
	std::mutex sem;
	int semCount = 1;
	std::vector<double> treesOOB;
	conf config;

	void threadFunction(TRForestModel& model, int num)
	{
		TDecisionTree dTree;
		std::vector<char> dataIdx(dataset.goals.size(),0);
		std::vector<int> testIdx;
		TDataset::TBaggingIterator it = dataset.BaggingIterator();
		int datasize = 0;
		std::vector<int> classCount(dataset.classCount.size(),0);

		it.ResetShuffle(num);
		it.SetLearnMode();
		int size = it.InstanceFoldNumbers.size();

		for (int i = 0; i < size; ++i) {
			if (it.InstanceFoldNumbers[i]) {
				dataIdx[i] = 1;
				++classCount[dataset.goals[i]];
				++datasize;
			}
			else
				testIdx.push_back(i);
		}

		int featureCount = dataset.featuresMatrix[0].size();
		std::vector<int> fIdx(featureCount, 0);
		for (int i = 0; i < featureCount; ++i) {
			fIdx[i] = i;
		}

		unsigned seed = std::chrono::system_clock::now().time_since_epoch() /
			std::chrono::milliseconds(1);
		shuffle(fIdx.begin(), fIdx.end(), std::default_random_engine(seed+num));
		if (config.featureSubset == "float") {
			fIdx.resize((int)((double)fIdx.size() * config.featureSubsetVal));
		}
		else if(config.featureSubset == "sqrt") {
			fIdx.resize((int)sqrt(fIdx.size()));
		}
		else if (config.featureSubset == "log") {
			fIdx.resize((int)log2(fIdx.size()));
		}
		dTree.ConstructTree(dataset, dataIdx, classCount, datasize, fIdx, config);
		g_lock.lock();
		model.forest.push_back(dTree);
		g_lock.unlock();

		if (config.OOB) {
			int cl;
			int wrong = 0;
			for (size_t idx : testIdx) {
				cl = dTree.Prediction<double>(dataset.featuresMatrix[idx]);
				if (cl != dataset.goals[idx])
					++wrong;
			}
			std::cout << "t" + std::to_string(num) + " OOB " + std::to_string((double)wrong / dataIdx.size()) + "\n";
		}
		sem.lock();
		++semCount;
		sem.unlock();
	}

	TRForestModel Solve(conf cf) {
		config = cf;
		semCount = config.threadNumber;
		dataset.SortFeatures();
		dataset.CalculateSplitpoints();
		dataset.PrepareGoals();

		TRForestModel model;
		std::cout << "Creating Trees:\n";
		for (int i = 0; i < config.treeCount; ++i) {
			while (true) {
				sem.lock();
				if (semCount > 0) {
					--semCount;
					std::cout << "t"+std::to_string(i)+"\n";
					std::thread(&TRForestSolver::threadFunction, this, std::ref(model), i).detach();
					sem.unlock();
					break;
				}
				else {
					sem.unlock();
					Sleep(500);
				}				
			}			
		}
		while (true) {
			std::cout << "wait...\n";
			Sleep(3000);
			g_lock.lock();
			if (model.forest.size() == config.treeCount)
				break;
			g_lock.unlock();
		}
		g_lock.unlock();
		auto it = dataset.classes.begin();
		while (it != dataset.classes.end()) {
			model.classTranslation.insert({it->second,it->first});
			++it;
		}
		return model;
	}
	
};