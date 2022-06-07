#ifndef SRC_HyperParams_H_
#define SRC_HyperParams_H_

#include "N3LDG.h"
#include "Options.h"

using namespace nr;
using namespace std;

struct HyperParams {
	dtype nnRegular; // for optimization
	dtype adaAlpha;  // for optimization
	dtype adaEps; // for optimization
	dtype dropProb;
	dtype clips;
	dtype delta;
	dtype oracleProb;

	int charContext;
	int charWindow;
	int hiddenSize;
	int rnnHiddenSize;
	int batch;
	int stateHiddenSize;
	int actionHiddenSize;
	int charHiddenSize;

	int wordDim;
	int charDim;
	int extWordDim;
	int actionDim;
	int etypeDim;
	int wordConcatDim;
	int stateConcatDim;
	int eduConcatDim;
	int eduHiddenDim;
	int tagDim;
	int synCombineDim;
	int encoderDim;

	int wordRepresentHiddenSize;
	int actionNum;
	string root;
	int rootID;
	int beam;
	int maxStateSize;
	int maxSentSize;
	int maxEDUSize;
	bool dynamicOracle;

	unordered_map<string, int> word_stat;
	unordered_map<string, int> tag_stat;
	unordered_map<string, int> action_stat;
	unordered_map<string, int> etype_stat;

	Alphabet wordAlpha;
	Alphabet actionAlpha;
	Alphabet labelAlpha;
	Alphabet tagAlpha;
	Alphabet etypeAlpha;

public:
	HyperParams() {
		bAssigned = false;
		dynamicOracle = false;
	}

	void write(std::ofstream &os){
		wordAlpha.write(os);
		actionAlpha.write(os);
		labelAlpha.write(os);
		tagAlpha.write(os);
		etypeAlpha.write(os);
	}

	void read(std::ifstream &is){
		wordAlpha.read(is);
		actionAlpha.read(is);
		labelAlpha.read(is);
		tagAlpha.read(is);
		etypeAlpha.read(is);
	}

	void setRequared(Options &opt) {
		bAssigned = true;
		nnRegular = opt.regParameter;
		adaAlpha = opt.adaAlpha;
		adaEps = opt.adaEps;
		dropProb = opt.dropProb;
		charContext = opt.charContext;
		hiddenSize = opt.hiddenSize;
		rnnHiddenSize = opt.rnnHiddenSize;
		stateHiddenSize = opt.stateHiddenSize;
		charHiddenSize = opt.charHiddenSize;
		wordRepresentHiddenSize = opt.wordRepresentHiddenSize;


		charWindow = charContext * 2 + 1;
		synCombineDim = 1200;
		eduHiddenDim = rnnHiddenSize * 4;
		encoderDim = rnnHiddenSize * 2;
		stateConcatDim = encoderDim * 4;
		oracleProb = opt.oracleProb;
		batch = opt.batchSize;
		clips = opt.clips;
		delta = opt.delta;
		beam = opt.beam;
		maxStateSize = opt.maxStateSize;
		maxSentSize = opt.maxSentSize;
		maxEDUSize = opt.maxEDUSize;
	}

	void clear() {
		bAssigned = false;
	}

	bool bValid() {
		return bAssigned;
	}

	void print() {}


private:
	bool bAssigned;
};

#endif /* HyperParams_H_ */
