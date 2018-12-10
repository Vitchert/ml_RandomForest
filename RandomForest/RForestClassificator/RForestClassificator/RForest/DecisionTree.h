#pragma once
#include <vector>
#include <fstream>
#include <numeric>
#include <map>
#include <chrono> 
#include <stack>
#include "Dataset.h"

struct TDecisionTreeNode {
	int weight = 0;
	int featureIndex = 0;
	double threshold = 0;
	double splitMetricValue = 0;
	int classIndex = -1;
	int objectCount = 0;
	int leftChildIndex = -1;
	int rightChildIndex = -1;
};

struct TDecisionTree {
	std::vector<TDecisionTreeNode> tree;
	int numberFeatures;
	bool eachNodeShuffle;
	int seed = 1;
	bool timeRandom = false;
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
				treeOut << node.weight << " " << node.featureIndex << " " << node.threshold << " " << node.splitMetricValue << " " << node.classIndex << " " << node.objectCount << " " << node.leftChildIndex << " " << node.rightChildIndex << " ";
			}
		}
	}
	static TDecisionTree LoadFromFile(std::ifstream& treeIn) {

		size_t treeSize;
		treeIn >> treeSize;


		TDecisionTree desicionTree;
		desicionTree.tree.resize(treeSize);

		for (size_t nodeIdx = 0; nodeIdx < treeSize; ++nodeIdx) {
			treeIn >> desicionTree.tree[nodeIdx].weight >> desicionTree.tree[nodeIdx].featureIndex >> desicionTree.tree[nodeIdx].threshold >> desicionTree.tree[nodeIdx].splitMetricValue >> desicionTree.tree[nodeIdx].classIndex >> desicionTree.tree[nodeIdx].objectCount >> desicionTree.tree[nodeIdx].leftChildIndex >> desicionTree.tree[nodeIdx].rightChildIndex;
		}

		return desicionTree;
	};

	void ConstructTree(TDataset& dataset, std::vector<int> dataIdx, std::vector<int> classCount,int dataIdxsize, std::vector<int> fIdx, conf config) {
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
		if (config.randomType == "time") {
			timeRandom = true;
		}
		else {
			if (config.randomType == "seed")
				seed = config.seed;
		}
		ConstructTreeFunc(dataset, dataIdx, classCount, fIdx, config.max_depth);
	}

	struct TBestSplit {
		int featureIdx;
		double splitVal;
		double splitMetricVal;
	};

	TBestSplit FindBestSplit(TDataset& dataset, const std::vector<int>& dataIdx, std::vector<int>& classCount, std::vector<int>& fIdx) {
		int dataIdxsize = dataIdx.size();
		if (eachNodeShuffle) {
			if (timeRandom) {
				seed = std::chrono::system_clock::now().time_since_epoch() / std::chrono::milliseconds(1);
			}
			shuffle(fIdx.begin(), fIdx.end(), std::default_random_engine(seed));
		}
		TBestSplit bestSplit;
		bestSplit.featureIdx = -1;
		bestSplit.splitVal = 0;
		bestSplit.splitMetricVal = -1;

		int classSize = classCount.size();
		int instanceCount = dataset.featuresMatrix.size();
		double minGini = 1;
		for (int i = 0; i < classSize; ++i)
			minGini -= ((double)(classCount[i] * classCount[i])) / (dataIdxsize*dataIdxsize);
		bestSplit.splitMetricVal = minGini;

		std::vector<int> leftClasses(classSize, 0);
		std::vector<int> rightClasses;
		for (int fi = 0; fi < numberFeatures; ++fi) {
			int featureIdx = fIdx[fi];
			std::vector<double> splitpoints = dataset.splitPointsMatrix[featureIdx];
			for (double split : splitpoints) {
				int leftsize = 0;
				int rightsize = dataIdxsize;
				rightClasses = classCount;
				std::fill(leftClasses.begin(), leftClasses.end(), 0);
				for (int dataPos : dataIdx) {
					if (dataset.featuresMatrix[dataPos][featureIdx] < split) {
						++leftsize;
						--rightsize;
						++leftClasses[dataset.goals[dataPos]];
						--rightClasses[dataset.goals[dataPos]];
					}
				}

				double leftGini = 1;
				double rightGini = 1;
				for (int i = 0; i < classSize; ++i) {
					leftGini -= ((double)(leftClasses[i] * leftClasses[i])) / (leftsize*leftsize);
					rightGini -= ((double)(rightClasses[i] * rightClasses[i])) / (rightsize*rightsize);
				}
				double Gini = (leftsize*leftGini + rightsize * rightGini) / dataIdxsize;
				if (Gini < minGini) {
					minGini = Gini;
					bestSplit.featureIdx = featureIdx;
					bestSplit.splitVal = split;
					bestSplit.splitMetricVal = minGini;
					if ((!leftGini) || (!rightGini))
						break;
				}
			}
		}
		return bestSplit;
	}

	void add_class_node(TBestSplit & split, std::vector<int>& dataIdx, std::vector<int>& classCount) {
		TDecisionTreeNode node;
		node.objectCount = dataIdx.size();
		node.classIndex = std::max_element(classCount.begin(), classCount.end()) - classCount.begin();
		node.splitMetricValue = split.splitMetricVal;
		tree.push_back(node);
	}

	void ConstructTreeFunc(TDataset& dataset, std::vector<int> dataIdx, std::vector<int> classCount, std::vector<int> fIdx, int max_depth) {
		std::stack<std::vector<int>> s_dataIdx;
		std::stack<std::vector<int>> s_classCount;
		std::stack<int> s_parentIdx;
		std::stack<int> s_depth;
		int cur_depth = 1;
		do {		
			TBestSplit split = FindBestSplit(dataset, dataIdx, classCount, fIdx);
			if (split.featureIdx < 0 || cur_depth >= max_depth) {
				add_class_node(split, dataIdx, classCount);
				if (s_dataIdx.size() > 0) {
					dataIdx = s_dataIdx.top();
					classCount = s_classCount.top();
					cur_depth = s_depth.top();
					tree[s_parentIdx.top()].rightChildIndex = tree.size();
					s_dataIdx.pop();
					s_classCount.pop();
					s_parentIdx.pop();
					s_depth.pop();
					continue;
				}
				else {
					break;
				}
			}
			else {
				int dataPos = 0;
				int size = dataset.goals.size();

				std::vector<int> dataIdxLeft;
				std::vector<int> dataIdxRight;
				dataIdxLeft.reserve(dataIdx.size());
				dataIdxRight.reserve(dataIdx.size());
				std::vector<int> leftClassCount(classCount.size(), 0);
				std::vector<int> rightClassCount(classCount.size(), 0);
				int leftDataSize = 0;
				int rightDataSize = 0;
				for (int dataPos : dataIdx) {
					if (dataset.featuresMatrix[dataPos][split.featureIdx] < split.splitVal) {
						++leftDataSize;
						++leftClassCount[dataset.goals[dataPos]];
						dataIdxLeft.push_back(dataPos);
					}
					else {
						++rightDataSize;
						++rightClassCount[dataset.goals[dataPos]];
						dataIdxRight.push_back(dataPos);
					}
				}

				TDecisionTreeNode node;
				node.objectCount = dataIdx.size();
				node.featureIndex = split.featureIdx;
				node.threshold = split.splitVal;
				node.splitMetricValue = split.splitMetricVal;
				tree.push_back(node);
				int curpos = tree.size() - 1;
				tree[curpos].leftChildIndex = tree.size();

				s_dataIdx.push(dataIdxRight);
				s_classCount.push(rightClassCount);
				s_parentIdx.push(curpos);
				s_depth.push(++cur_depth);
				dataIdx = dataIdxLeft;
				classCount = leftClassCount;
			}
		} while (true);
	}


};