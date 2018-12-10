#pragma once
#include <iostream>

struct conf {
	std::string mode = "";
	std::string modelPath = "";
	std::string predictionPath = "";
	int treeCount = 1;
	int threadNumber = 1;
	bool OOB = false;
	std::string maxNodeFeatures = "sqrt";
	double maxNodeFeaturesVal = 1;
	std::string featureSubset = "all";
	double featureSubsetVal = 1;
	bool shuffle_features = false;
	int rounds = 1;
	int folds = 2;
	std::string randomType = "seed";
	int seed = 1;
	std::string featuresFilePath = "";
	int max_depth = 3;

	static int ParseArguments(conf& config, int argc,const char *argv[])
	{
		for (int i = 1; i < argc; i++)
		{
			if (stricmp(argv[i], "-modelPath") == 0)
			{
				if (++i >= argc)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
				config.modelPath = argv[i];
			}

			if (stricmp(argv[i], "-featuresPath") == 0)
			{
				if (++i >= argc)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
				config.featuresFilePath = argv[i];
			}

			if (stricmp(argv[i], "-predictionPath") == 0)
			{
				if (++i >= argc)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
				config.predictionPath = argv[i];
			}

			else if (stricmp(argv[i], "-mode") == 0)
			{
				if (++i >= argc)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
				if ((stricmp(argv[i], "cv") == 0) || (stricmp(argv[i], "learn") == 0) || (stricmp(argv[i], "predict") == 0)) {
					config.mode = argv[i];
					if (stricmp(argv[i], "cv") == 0) {
						if (++i >= argc || sscanf(argv[i], "%d", &config.rounds) != 1)
						{
							printf("invalid parameter for %s\n", argv[i - 1]);
							return 1;
						}
						if (++i >= argc || sscanf(argv[i], "%d", &config.folds) != 1)
						{
							printf("invalid parameter for %s\n", argv[i - 2]);
							return 1;
						}
					}
				}
				else {
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}

			}

			else if (stricmp(argv[i], "-treeCount") == 0)
			{
				if (++i >= argc || sscanf(argv[i], "%d", &config.treeCount) != 1)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
			}

			else if (stricmp(argv[i], "-threadCount") == 0)
			{
				if (++i >= argc || sscanf(argv[i], "%d", &config.threadNumber) != 1)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
			}

			else if (stricmp(argv[i], "-oob") == 0)
			{
				config.OOB = true;
			}

			else if (stricmp(argv[i], "-shuffle") == 0)
			{
				config.shuffle_features = true;
			}

			else if (stricmp(argv[i], "-maxNodeFeatures") == 0)
			{
				if (++i >= argc)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
				if ((stricmp(argv[i], "sqrt") == 0) || (stricmp(argv[i], "log") == 0) || (stricmp(argv[i], "all") == 0) || (stricmp(argv[i], "float") == 0)) {
					config.maxNodeFeatures = argv[i];
					if (stricmp(argv[i], "float") == 0) {
						if (++i >= argc || sscanf(argv[i], "%f", &config.maxNodeFeaturesVal) != 1)
						{
							printf("invalid parameter for %s\n", argv[i - 1]);
							return 1;
						}
					}
				}
				else {
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
			}

			else if (stricmp(argv[i], "-featureSubset") == 0)
			{
				if (++i >= argc)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
				if ((stricmp(argv[i], "sqrt") == 0) || (stricmp(argv[i], "log") == 0) || (stricmp(argv[i], "all") == 0) || (stricmp(argv[i], "float") == 0)) {
					config.featureSubset = argv[i];
					if (stricmp(argv[i], "float") == 0) {
						if (++i >= argc || sscanf(argv[i], "%f", &config.featureSubsetVal) != 1)
						{
							printf("invalid parameter for %s\n", argv[i - 1]);
							return 1;
						}
					}
				}
				else {
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
			}

			else if (stricmp(argv[i], "-random") == 0)
			{
				if (++i >= argc)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
				if ((stricmp(argv[i], "seed") == 0) || (stricmp(argv[i], "time") == 0) ) {
					config.randomType = argv[i];
					if (stricmp(argv[i], "seed") == 0) {
						if (++i >= argc || sscanf(argv[i], "%d", &config.seed) != 1)
						{
							printf("invalid parameter for %s\n", argv[i - 1]);
							return 1;
						}
					}
				}
				else {
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
			}

			else if (stricmp(argv[i], "-depth") == 0)
			{
				if (++i >= argc || sscanf(argv[i], "%d", &config.max_depth) != 1)
				{
					printf("invalid parameter for %s\n", argv[i - 1]);
					return 1;
				}
			}
		}
		return 0;
	}
};