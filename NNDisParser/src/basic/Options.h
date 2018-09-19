#ifndef BASIC_OPTIONS_H
#define BASIC_OPTIONS_H

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include "N3LDG.h"

class Options {
public:
  int wordCutOff;
	int wordEmbSize;
	string wordEmbFile;
	bool wordFineTune;

	int tagEmbSize;
	bool tagFineTune;

	int etypeEmbSize;
	bool etypeFineTune;

	string charEmbFile;
	int charCutOff;
	int charEmbSize;
	int charHiddenSize;
	bool charFineTune;
	int charContext;

	dtype p;
	int stateHiddenSize;
	int wordRepresentHiddenSize;
	int hiddenSize;
	int rnnHiddenSize;

  int maxIter;
  int batchSize;

	int maxSentSize;
	int maxEDUSize;
	int maxStateSize;

  dtype adaEps;
  dtype adaAlpha;
  dtype regParameter;
  dtype dropProb;
	dtype clips;
	dtype delta;

  int verboseIter;
  bool saveIntermediate;
  int maxInstance;
  vector<string> testFiles;
  string outBest;

  int unkStrategy;
	int beam;
	int startBeam;

	dtype decay;
	int startDecay;

	int reachDrop;
	
	string conllFolder;
	string dumpFolder;

	dtype oracleProb;
	int startDynamicOracle;
	dtype oracleDecay;
  Options() {
    wordCutOff = 0;
    wordEmbSize = 200;
		wordEmbFile = "";
		wordFineTune = true;

		tagEmbSize = 200;
		tagFineTune = true;
		
		etypeEmbSize = 100;
		etypeFineTune = true;

		charCutOff = 0;
		charEmbSize = 50;
		charHiddenSize = 100;
		charEmbFile = "";
		charFineTune = true;
		charContext = 2;

		hiddenSize = 200;
		rnnHiddenSize = 200;
		stateHiddenSize = 200;
		wordRepresentHiddenSize = 200;

    maxIter = 1000;
    batchSize = 1;

    adaEps = 1e-6;
    adaAlpha = 0.001;
    regParameter = 1e-8;
    dropProb = -1;
		clips = 10;
		delta = 0;
		p = 0.1;

    verboseIter = 100;
    saveIntermediate = true;
    maxInstance = -1;
    testFiles.clear();
    outBest = "";
    unkStrategy = 1;
		beam = 1;
		startBeam = 0;


		decay = 0;
		startDecay = 0;

		oracleProb = 0.85;
		startDynamicOracle = 0;
		oracleDecay = 0.95;

		reachDrop = -1;

		maxSentSize = 128;
		maxEDUSize = 512;
		maxStateSize = 1024;
		dumpFolder = "";
		conllFolder = "";
  }

  virtual ~Options() {

  }

  void setOptions(const vector<string> &vecOption) {
    int i = 0;
    for (; i < vecOption.size(); ++i) {
      pair<string, string> pr;
      string2pair(vecOption[i], pr, '=');
      if (pr.first == "wordCutOff")
        wordCutOff = atoi(pr.second.c_str());
      if (pr.first == "wordEmbSize")
        wordEmbSize = atoi(pr.second.c_str());
      if (pr.first == "wordEmbFile")
        wordEmbFile = pr.second;
      if (pr.first == "wordFineTune")
				wordFineTune = (pr.second == "true") ? true : false;

      if (pr.first == "charCutOff")
        charCutOff = atoi(pr.second.c_str());
      if (pr.first == "charEmbSize")
        charEmbSize = atoi(pr.second.c_str());
      if (pr.first == "charEmbFile")
        charEmbFile = pr.second;
      if (pr.first == "charFineTune")
				charFineTune = (pr.second == "true") ? true : false;
      if (pr.first == "charHiddenSize")
        charHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "charContext")
				charContext = atoi(pr.second.c_str());

      if (pr.first == "maxIter")
        maxIter = atoi(pr.second.c_str());
      if (pr.first == "batchSize")
        batchSize = atoi(pr.second.c_str());
      if (pr.first == "adaEps")
        adaEps = atof(pr.second.c_str());
      if (pr.first == "adaAlpha")
        adaAlpha = atof(pr.second.c_str());
      if (pr.first == "regParameter")
        regParameter = atof(pr.second.c_str());
      if (pr.first == "dropProb")
        dropProb = atof(pr.second.c_str());
      if (pr.first == "clips")
        clips = atof(pr.second.c_str());
      if (pr.first == "delta")
        delta = atof(pr.second.c_str());
      if (pr.first == "p")
				p = atof(pr.second.c_str());

      if (pr.first == "tagEmbSize")
        tagEmbSize = atoi(pr.second.c_str());
      if (pr.first == "tagFineTune")
				tagFineTune = (pr.second == "true") ? true : false;

      if (pr.first == "etypeEmbSize")
        etypeEmbSize = atoi(pr.second.c_str());
      if (pr.first == "etypeFineTune")
				etypeFineTune = (pr.second == "true") ? true : false;

      if (pr.first == "stateHiddenSize")
        stateHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "wordRepresentHiddenSize")
        wordRepresentHiddenSize = atoi(pr.second.c_str());
      if (pr.first == "hiddenSize")
        hiddenSize = atoi(pr.second.c_str());
      if (pr.first == "rnnHiddenSize")
        rnnHiddenSize = atoi(pr.second.c_str());

      if (pr.first == "verboseIter")
        verboseIter = atoi(pr.second.c_str());
      if (pr.first == "saveIntermediate")
        saveIntermediate = (pr.second == "true") ? true : false;

      if (pr.first == "maxInstance")
        maxInstance = atoi(pr.second.c_str());
      if (pr.first == "testFile")
        testFiles.push_back(pr.second);
      if (pr.first == "outBest")
        outBest = pr.second;
      if (pr.first == "beam")
        beam = atoi(pr.second.c_str());
      if (pr.first == "startBeam")
        startBeam = atoi(pr.second.c_str());

      if (pr.first == "maxSentSize")
        maxSentSize = atoi(pr.second.c_str());
      if (pr.first == "maxEDUSize")
        maxEDUSize = atoi(pr.second.c_str());
      if (pr.first == "maxStateSize")
        maxStateSize = atoi(pr.second.c_str());

      if (pr.first == "decay")
        decay = atof(pr.second.c_str());

      if (pr.first == "startDecay")
        startDecay = atoi(pr.second.c_str());

      if (pr.first == "reachDrop")
        reachDrop = atoi(pr.second.c_str());


      if (pr.first == "oracleDecay")
				oracleDecay = atof(pr.second.c_str());
      if (pr.first == "oracleProb")
				oracleProb = atof(pr.second.c_str());
      if (pr.first == "startDynamicOracle")
        startDynamicOracle = atoi(pr.second.c_str());

      if (pr.first == "conllFolder")
        conllFolder = pr.second;

      if (pr.first == "dumpFolder")
        dumpFolder = pr.second;
    }
  }

  void showOptions() {
    std::cout << "maxIter = " << maxIter << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "adaEps = " << adaEps << std::endl;
    std::cout << "adaAlpha = " << adaAlpha << std::endl;
    std::cout << "regParameter = " << regParameter << std::endl;
    std::cout << "dropProb = " << dropProb << std::endl;
    std::cout << "clips = " << clips << std::endl;
    std::cout << "delta = " << delta << std::endl;
    std::cout << "p = " << p << std::endl;

    std::cout << "hiddenSize = " << hiddenSize << std::endl;
    std::cout << "rnnHiddenSize = " << rnnHiddenSize << std::endl;
    std::cout << "stateHiddenSize = " << stateHiddenSize << std::endl;
    std::cout << "wordRepresentHiddenSize = " << wordRepresentHiddenSize << std::endl;

    std::cout << "wordCutOff = " << wordCutOff << std::endl;
    std::cout << "wordEmbSize = " << wordEmbSize << std::endl;
    std::cout << "wordEmbFile = " << wordEmbFile << std::endl;
    std::cout << "wordFineTune = " << wordFineTune << std::endl;

    std::cout << "charCutOff = " << charCutOff << std::endl;
    std::cout << "charEmbSize = " << charEmbSize << std::endl;
    std::cout << "charEmbFile = " << charEmbFile << std::endl;
    std::cout << "charFineTune = " << charFineTune << std::endl;
		std::cout << "charHiddenSize = " << charHiddenSize << std::endl;
		std::cout << "charContext = " << charContext << std::endl;

    std::cout << "tagEmbSize = " << tagEmbSize << std::endl;
    std::cout << "tagFineTune = " << tagFineTune << std::endl;

    std::cout << "etypeEmbSize = " << etypeEmbSize << std::endl;
    std::cout << "etypeFineTune = " << etypeFineTune << std::endl;

    std::cout << "unkStrategy = " << unkStrategy << std::endl;

    std::cout << "verboseIter = " << verboseIter << std::endl;
    std::cout << "saveIntermediate = " << saveIntermediate << std::endl;
    std::cout << "maxInstance = " << maxInstance << std::endl;
    for (int idx = 0; idx < testFiles.size(); idx++) {
      std::cout << "testFile = " << testFiles[idx] << std::endl;
    }
    std::cout << "outBest = " << outBest << std::endl;
    std::cout << "beam = " << beam << std::endl;
    std::cout << "startBeam = " << startBeam << std::endl;

    std::cout << "maxSentSize = " << maxSentSize << std::endl;
    std::cout << "maxEDUSize = " << maxEDUSize << std::endl;
    std::cout << "maxStateSize = " << maxStateSize << std::endl;

    std::cout << "startDecay = " << startDecay << std::endl;
    std::cout << "decay = " << decay << std::endl;
    std::cout << "reachDrop = " << reachDrop << std::endl;
    std::cout << "dumpFolder = " << dumpFolder << std::endl;
    std::cout << "conllFolder = " << conllFolder << std::endl;
    std::cout << "oracleProb = " << oracleProb << std::endl;
    std::cout << "oracleDecay = " << oracleDecay << std::endl;
    std::cout << "startDynamicOracle = " << startDynamicOracle << std::endl;


    std:cout << std::endl;
  }

  void load(const std::string &infile) {
    ifstream inf;
    inf.open(infile.c_str());
    vector<string> vecLine;
    while (1) {
      string strLine;
      if (!my_getline(inf, strLine)) {
        break;
      }
      if (strLine.empty())
        continue;
      vecLine.push_back(strLine);
    }
    inf.close();
    setOptions(vecLine);
  }
};

#endif
