#include <iostream>
#include "RForestSolver.h"


int main(int argc, const char** argv) {
	(void)argc;
	conf config;
	if (conf::ParseArguments(config, argc, argv)) {
		return 1;
	}

	TDataset dataset;
	dataset.ReadFromFile(config.featuresFilePath);

	if (config.mode == "learn") {
		TRForestSolver solver;
		solver.AddDataset(dataset);
		TRForestModel model = solver.Solve(config);
		model.SaveToFile(config.modelPath);
	}
	else if (config.mode == "predict") {
		std::ofstream predictionsOut(config.predictionPath);
		if (!predictionsOut.is_open()) {
			std::cout << "Cant create predictions file in path=\"" << config.predictionPath << "\" \n";
			exit(EXIT_FAILURE);
		}
		predictionsOut.precision(20);

		TRForestModel model = TRForestModel::LoadFromFile(config.modelPath);
		bool translateClasses = true;
		for (std::vector<std::vector<double>>::iterator it = dataset.featuresMatrix.begin(); it != dataset.featuresMatrix.end(); ++it) {
			predictionsOut << model.Prediction(*it, translateClasses) << "\n";
		}
	}
	else if (config.mode == "cv") {
		TInstance instance;
		TDataset::TCVIterator iterator = dataset.CrossValidationIterator(config.folds);
		double ACCmean = 0;
		int prediction;
		for (int i = 0; i < config.rounds; ++i) {
			std::cout << "CV round " << i << std::endl;

			iterator.ResetShuffle();
			double roundACC = 0;
			int testsize = 1;
			for (int j = 0; j < config.folds; ++j) {
				iterator.SetLearnMode();
				iterator.SetTestFold(j);
				TRForestSolver solver;
				while (iterator.IsValid()) {
					instance = *iterator;
					solver.Add(instance.Features, instance.Goal, instance.Weight);
					++iterator;
				}

				TRForestModel model = solver.Solve(config);
				iterator.SetTestMode();

				double wrongPredCnt = 0;
				int predictionsize = 0;
				while (iterator.IsValid()) {
					instance = *iterator;
					prediction = model.Prediction(instance.Features,true);
					if (prediction != instance.Goal)
						wrongPredCnt++;
					++iterator;
					++predictionsize;
				}
				double foldACC = 0;
				if (predictionsize)
					foldACC = (predictionsize - wrongPredCnt) / predictionsize;
				if (foldACC != 0) {
					roundACC += (foldACC - roundACC) / (testsize++);
				}
				else {					
					testsize++;
				}
				std::cout << "fold " << j << " acc= " << foldACC << std::endl;
				std::cout << "correct " << predictionsize - (int)wrongPredCnt << " wrong " << (int)wrongPredCnt << std::endl;
			}
			std::cout << "CV round " << i <<" accuracy: " << roundACC << std::endl;
			ACCmean += (roundACC - ACCmean) / (i + 1);
		}
		std::cout << std::endl << "Mean accuracy: " << ACCmean << std::endl;
	}
	return 0;
}

