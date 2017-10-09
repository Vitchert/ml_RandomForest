#pragma once
#include <vector>
#include <fstream>
#include <numeric>
#include <map>
#include <chrono> 
#include "Dataset.h"

struct TDecisionTreeNode {
	int weight = 0;
	int featureIndex = 0;
	double threshold = 0;
	int classIndex = -1;
	int leftChildIndex = -1;
	int rightChildIndex = -1;
};

struct TDecisionTree {
	std::vector<TDecisionTreeNode> tree;
	int numberFeatures;
	bool eachNodeShuffle;
	template <typename T>
	int Prediction(const std::vector<T>& features) const {
		int idx = 0;
		while (tree[idx].classIndex < 0) {
			idx = features[tree[idx].featureIndex] < tree[idx].threshold ? tree[idx].leftChildIndex : tree[idx].rightChildIndex;
		}
		return tree[idx].classIndex;
	}
	void SaveToFile(std::ofstream& treeOut) {
		if (tree.size()) {
			treeOut << tree.size() << " ";
			for (TDecisionTreeNode node : tree) {
				treeOut << node.weight << " " << node.featureIndex << " " << node.threshold << " " << node.classIndex << " " << node.leftChildIndex << " " << node.rightChildIndex << " ";
			}
		}
	}
	static TDecisionTree LoadFromFile(std::ifstream& treeIn) {

		size_t treeSize;
		treeIn >> treeSize;


		TDecisionTree desicionTree;
		desicionTree.tree.resize(treeSize);

		for (size_t nodeIdx = 0; nodeIdx < treeSize; ++treeSize) {
			treeIn >> desicionTree.tree[nodeIdx].weight >> desicionTree.tree[nodeIdx].featureIndex >> desicionTree.tree[nodeIdx].threshold >> desicionTree.tree[nodeIdx].classIndex >> desicionTree.tree[nodeIdx].leftChildIndex >> desicionTree.tree[nodeIdx].rightChildIndex;
		}

		return desicionTree;
	};

	void ConstructTree(TDataset& dataset, std::vector<char> dataIdx, std::vector<int> classCount,int dataIdxsize, std::vector<int> fIdx, conf config) {
		if (config.maxNodeFeatures == "float") {
			numberFeatures = (int)((double)fIdx.size() * config.maxNodeFeaturesVal);
		}
		else if (config.maxNodeFeatures == "sqrt") {
			numberFeatures = (int)sqrt(fIdx.size());
		}
		else if (config.maxNodeFeatures == "log") {
			numberFeatures = (int)log2(fIdx.size());
		}
		else if (config.maxNodeFeatures == "all") {
			numberFeatures = fIdx.size();
		}
		eachNodeShuffle = config.shuffle_features;
		ConstructTreeRecursion(dataset, dataIdx, classCount, dataIdxsize, fIdx);
	}

	void ConstructTreeRecursion(TDataset& dataset, std::vector<char> dataIdx, std::vector<int> classCount, int dataIdxsize, std::vector<int> fIdx) {
		TDecisionTreeNode node;
		TBestSplit split = FindBestSplit(dataset, dataIdx, classCount, dataIdxsize, fIdx);
		if (split.featureIdx < 0) {
			node.classIndex = std::max_element(classCount.begin(), classCount.end()) - classCount.begin();
			tree.push_back(node);
		}
		else {
			int dataPos = 0;
			int size = dataset.goals.size();

			std::vector<char> dataIdxLeft(dataIdx.size(), 0);
			std::vector<char>& dataIdxRight = dataIdx;
			std::vector<int> leftClassCount(classCount.size(), 0);
			std::vector<int>& rightClassCount = classCount;
			int leftDataSize = 0;
			int rightDataSize = dataIdxsize;
			while ((dataPos < size) && (dataIdx[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]] != 1)) {
				++dataPos;
			}
			while ((dataPos < size) && (dataset.featuresMatrix[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]][split.featureIdx] < split.splitVal)) {
				dataIdxRight[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]] = 0;
				dataIdxLeft[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]] = 1;
				--rightClassCount[dataset.goals[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]]];
				++leftClassCount[dataset.goals[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]]];
				--rightDataSize;
				++leftDataSize;

				++dataPos;
				while ((dataPos < size) && (dataIdx[dataset.sortedByIdxFeaturesMatrix[split.featureIdx][dataPos]] != 1)) {
					++dataPos;
				}
			}

			node.featureIndex = split.featureIdx;
			node.threshold = split.splitVal;
			tree.push_back(node);
			int curpos = tree.size() - 1;
			tree[curpos].leftChildIndex = tree.size();
			ConstructTreeRecursion(dataset, dataIdxLeft, leftClassCount, leftDataSize, fIdx);
			tree[curpos].rightChildIndex = tree.size();
			ConstructTreeRecursion(dataset, dataIdxRight, rightClassCount, rightDataSize, fIdx);
		}
	}

	struct TBestSplit {
		int featureIdx;
		double splitVal;
	};

	TBestSplit FindBestSplit(TDataset& dataset,const std::vector<char>& dataIdx, std::vector<int>& classCount,int dataIdxsize, std::vector<int>& fIdx) {
		
#ifdef EACH_NODE_SHUFFLE
		unsigned seed = std::chrono::system_clock::now().time_since_epoch() /
			std::chrono::milliseconds(1);
		shuffle(fIdx.begin(), fIdx.end(), std::default_random_engine(seed));
#endif
		TBestSplit bestSplit;
		bestSplit.featureIdx = -1;
		bestSplit.splitVal = 0;

		double minGini = 1;
		int classSize = classCount.size();
		int instanceCount = dataset.featuresMatrix.size();
		for (int i = 0; i < classSize; ++i) 
			minGini -= (double)(classCount[i] * classCount[i]) / (dataIdxsize*dataIdxsize);

		std::vector<int> leftClasses(classSize,0);
		std::vector<int> rightClasses;
		for (int fi = 0; fi < numberFeatures; ++fi) {
			int featureIdx = fIdx[fi];
			std::vector<double> splitpoints = dataset.splitPointsMatrix[featureIdx];
			int dataPos = 0;
			int leftsize = 0;
			int rightsize = dataIdxsize;
			rightClasses = classCount;
			std::fill(leftClasses.begin(), leftClasses.end(),0);
			while ((dataPos < instanceCount) && (dataIdx[dataset.sortedByIdxFeaturesMatrix[featureIdx][dataPos]] != 1))
				++dataPos;

			for (double split : splitpoints) {
				while ((dataPos < instanceCount) && (dataset.featuresMatrix[dataset.sortedByIdxFeaturesMatrix[featureIdx][dataPos]][featureIdx] < split)) {
					++leftsize;
					--rightsize;
					++leftClasses[dataset.goals[dataset.sortedByIdxFeaturesMatrix[featureIdx][dataPos]]];
					--rightClasses[dataset.goals[dataset.sortedByIdxFeaturesMatrix[featureIdx][dataPos]]];
					do {
						++dataPos;
					} while ((dataPos < instanceCount) && (dataIdx[dataset.sortedByIdxFeaturesMatrix[featureIdx][dataPos]] != 1));
				}

				double leftGini = 1;
				double rightGini = 1;
				for (int i = 0; i < classSize; ++i) {
					leftGini -= (double)(leftClasses[i] * leftClasses[i]) / (leftsize*leftsize);
					rightGini -= (double)(rightClasses[i] * rightClasses[i]) / (rightsize*rightsize);
				}
				double Gini = (leftGini + rightGini) / 2.;
				if (Gini < minGini) {
					minGini = Gini;
					bestSplit.featureIdx = featureIdx;
					bestSplit.splitVal = split;
					if ((!leftGini) || (!rightGini))
						break;
				}
			}
		}
		return bestSplit;
	}
	
};