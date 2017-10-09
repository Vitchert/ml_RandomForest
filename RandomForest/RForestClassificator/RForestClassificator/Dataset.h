#pragma once

#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include <map>
#include "Conf.h"

struct TInstance {
	std::vector<double> Features;
	double Goal;
	double Weight;
};
struct TDataset {
	std::vector<std::vector<double>> featuresMatrix;
	std::vector<double> goals;
	std::vector<double> weights;
	int FeaturesCount;
	std::vector<int> classCount;
	std::map<double, double> classes;
	std::vector<std::vector<int>> sortedByIdxFeaturesMatrix;
	std::vector<std::vector<double>> splitPointsMatrix;
	
	void SortFeatures() {
		std::vector<int> fIdx(featuresMatrix.size());
		std::iota(std::begin(fIdx), std::end(fIdx), 0);
		int fSize = featuresMatrix[0].size();
		for (int i = 0; i < fSize; ++i) {
			SortByFeatureIdx(fIdx, i);
			sortedByIdxFeaturesMatrix.push_back(fIdx);
		}
		
	}

	void SortByFeatureIdx(std::vector<int>& dataIdx, int i) {
		std::sort(
			std::begin(dataIdx), std::end(dataIdx),
			[&](size_t a, size_t b) { return featuresMatrix[a][i] <featuresMatrix[b][i]; }
		);

	}

	void CalculateSplitpoints() {
		int instanceCount = featuresMatrix.size();
		std::vector<double> splits;
		int FeaturesCount = featuresMatrix[0].size();
		for (int j = 0; j < FeaturesCount; ++j) {
			int i = 1;
			double lastClass = goals[sortedByIdxFeaturesMatrix[j][0]];
			double lastVal = featuresMatrix[sortedByIdxFeaturesMatrix[j][0]][j];
			double newVal = lastVal;
			while (i < instanceCount) {
				if (goals[sortedByIdxFeaturesMatrix[j][i]] == lastClass) {
					++i;
					continue;
				}
				newVal = featuresMatrix[sortedByIdxFeaturesMatrix[j][i]][j];
				if (newVal != lastVal)
					splits.push_back((lastVal + featuresMatrix[sortedByIdxFeaturesMatrix[j][i]][j]) / 2.);
				lastVal = newVal;
				lastClass = goals[sortedByIdxFeaturesMatrix[j][i]];
			}
			splitPointsMatrix.push_back( splits);
			splits.clear();
		}		
	}
	
	void ParseFirst(const std::string& str) {
		std::stringstream featuresStream(str);
		std::vector<double> featureset;
		double feature;
		std::string queryId, url;
		int FeatureCount = 0;

		featuresStream >> feature; //get goal
		goals.push_back(feature);

		weights.push_back(1);

		while (featuresStream >> feature) {
			featureset.push_back(feature);
			FeatureCount++;
		}
		FeaturesCount = FeatureCount;
		featuresMatrix.push_back(featureset);
	};

	void Parse(const std::string& str) {
		std::stringstream featuresStream(str);
		std::vector<double> featureset(FeaturesCount);
		double feature;
		std::string queryId, url;
		int FeatureNumber = 0;

		featuresStream >> feature; //get goal
		goals.push_back(feature);

		weights.push_back(1);

		while (featuresStream >> feature) {
			featureset[FeatureNumber++] = feature;
		}
		featuresMatrix.push_back(featureset);
	};
	
	void PrepareGoals() {
		
		int size = goals.size();
		for (int i = 0; i < size; ++i) {
			if (!classes.count(goals[i])) {
				classes.insert({ goals[i] , classes.size() });
			}
		}
		classCount.resize(classes.size(), 0);
		for (int i = 0; i < size; ++i) {
			goals[i] = classes[goals[i]];
			++classCount[goals[i]];
		}
	}
	
	void ReadFromFile(const std::string& featuresPath) {
		std::ifstream featuresIn(featuresPath);
		if (!featuresIn.is_open()) {
			std::cout << "Cant open dataset file\n";
			exit(EXIT_FAILURE);
		}
		std::string featuresString;
		if (getline(featuresIn, featuresString)) 
			ParseFirst(featuresString);
		while (getline(featuresIn, featuresString))
		{
			if (featuresString.empty())
				continue;
			Parse(featuresString);
		}
	};

	enum EIteratorType {
		LearnIterator,
		TestIterator,
	};

	class TCVIterator {
	private:
		const TDataset& ParentDataset;

		TInstance Instance;

		size_t FoldsCount;

		EIteratorType IteratorType;
		size_t TestFoldNumber;

		std::vector<size_t> InstanceFoldNumbers;
		std::vector<size_t>::const_iterator Current;

		std::mt19937 RandomGenerator;
	public:
		TCVIterator(const TDataset& ParentDataset,
			const size_t foldsCount,
			const EIteratorType iteratorType) 
			: ParentDataset(ParentDataset)
			, FoldsCount(foldsCount)
			, IteratorType(iteratorType)
			, InstanceFoldNumbers(ParentDataset.featuresMatrix.size())
		{
		}

		void ResetShuffle() {
			std::vector<size_t> instanceNumbers(ParentDataset.featuresMatrix.size());
			for (size_t instanceNumber = 0; instanceNumber < ParentDataset.featuresMatrix.size(); ++instanceNumber) {
				instanceNumbers[instanceNumber] = instanceNumber;
			}
			shuffle(instanceNumbers.begin(), instanceNumbers.end(), RandomGenerator);

			for (size_t instancePosition = 0; instancePosition < ParentDataset.featuresMatrix.size(); ++instancePosition) {
				InstanceFoldNumbers[instanceNumbers[instancePosition]] = instancePosition % FoldsCount;
			}
			Current = InstanceFoldNumbers.begin();
		}

		void SetTestFold(const size_t testFoldNumber) {
			TestFoldNumber = testFoldNumber;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}

		bool IsValid() const {
			return Current != InstanceFoldNumbers.end();
		}
		void SetTestMode() {
			IteratorType = TestIterator;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}
		void SetLearnMode() {
			IteratorType = LearnIterator;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}
		const TInstance& operator * () {
			Instance.Features = ParentDataset.featuresMatrix[Current - InstanceFoldNumbers.begin()];
			Instance.Goal = ParentDataset.goals[Current - InstanceFoldNumbers.begin()];
			Instance.Weight = ParentDataset.weights[Current - InstanceFoldNumbers.begin()];
			return Instance;
		}
		const TInstance* operator ->() {
			Instance.Features = ParentDataset.featuresMatrix[Current - InstanceFoldNumbers.begin()];
			Instance.Goal = ParentDataset.goals[Current - InstanceFoldNumbers.begin()];
			Instance.Weight = ParentDataset.weights[Current - InstanceFoldNumbers.begin()];
			return &Instance;
		}
		TDataset::TCVIterator& operator++() {
			Advance();
			return *this;
		}
	private:
		void Advance() {
			while (IsValid()) {
				++Current;
				if (IsValid() && TakeCurrent()) {
					break;
				}
			}
		}
		bool TakeCurrent() const {
			switch (IteratorType) {
			case LearnIterator: return *Current != TestFoldNumber;
			case TestIterator: return *Current == TestFoldNumber;
			}
			return false;
		}
	};

	class TBaggingIterator {
	private:
		const TDataset& ParentDataset;

		TInstance Instance;

		EIteratorType IteratorType;

		//vector<size_t> InstanceFoldNumbers;
		std::vector<size_t>::const_iterator Current;

		std::mt19937 RandomGenerator;
	public:
		std::vector<size_t> InstanceFoldNumbers;
		TBaggingIterator(const TDataset& ParentDataset,
			const EIteratorType iteratorType)
			: ParentDataset(ParentDataset)
			, IteratorType(iteratorType)
			, InstanceFoldNumbers(ParentDataset.featuresMatrix.size())
		{
		}

		void ResetShuffle(size_t seed) {
			RandomGenerator.seed(seed);
			int size = ParentDataset.featuresMatrix.size();
			std::vector<size_t> instanceNumbers(size,0); //TEST = 0
	
			for (size_t instanceNumber = 0; instanceNumber < size; ++instanceNumber) {
				instanceNumbers[RandomGenerator() % size] = 1; //LEARN = 1
			}
			InstanceFoldNumbers = instanceNumbers;
			Current = InstanceFoldNumbers.begin();
		}

		bool IsValid() const {
			return Current != InstanceFoldNumbers.end();
		}
		void SetTestMode() {
			IteratorType = TestIterator;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}
		void SetLearnMode() {
			IteratorType = LearnIterator;
			Current = InstanceFoldNumbers.begin();
			Advance();
		}
		const TInstance& operator * () {
			Instance.Features = ParentDataset.featuresMatrix[Current - InstanceFoldNumbers.begin()];
			Instance.Goal = ParentDataset.goals[Current - InstanceFoldNumbers.begin()];
			Instance.Weight = ParentDataset.weights[Current - InstanceFoldNumbers.begin()];
			return Instance;
		}
		const TInstance* operator ->() {
			Instance.Features = ParentDataset.featuresMatrix[Current - InstanceFoldNumbers.begin()];
			Instance.Goal = ParentDataset.goals[Current - InstanceFoldNumbers.begin()];
			Instance.Weight = ParentDataset.weights[Current - InstanceFoldNumbers.begin()];
			return &Instance;
		}
		TDataset::TBaggingIterator& operator++() {
			Advance();
			return *this;
		}
	private:
		void Advance() {
			while (IsValid()) {
				++Current;
				if (IsValid() && TakeCurrent()) {
					break;
				}
			}
		}
		bool TakeCurrent() const {
			switch (IteratorType) {
			case LearnIterator: return *Current == 1;
			case TestIterator: return *Current == 0;
			}
			return false;
		}
	};

	TCVIterator CrossValidationIterator(const size_t foldsCount, const EIteratorType iteratorType = LearnIterator) const {
		return TCVIterator(*this, foldsCount, iteratorType);
	};
	TBaggingIterator BaggingIterator( const EIteratorType iteratorType = LearnIterator) const {
		return TBaggingIterator(*this, iteratorType);
	};

};