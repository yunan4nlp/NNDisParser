#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#include "HyperParams.h"
#include "EDUNodes.h"
#include "SynNodes.h"


class ModelParams {
public:
	SynParams syn_params;
	EDUParams edu_params;

	LookupTable etype_table;

	LSTM1Params edu_lstm_left_layer1_params;
	LSTM1Params edu_lstm_right_layer1_params;

	ActionParams scored_action_table;



public:
	inline bool initial(HyperParams &opts) {
		edu_params.initial(opts);
		syn_params.initial(opts);
		edu_lstm_left_layer1_params.initial(opts.rnnHiddenSize, opts.eduConcatDim);
		edu_lstm_right_layer1_params.initial(opts.rnnHiddenSize, opts.eduConcatDim);
		scored_action_table.initial(&opts.actionAlpha, opts.stateConcatDim);

		random_device rd;
		mt19937 gen(rd());
		gen.seed(0);
		std::normal_distribution<> d(0, 1);
		for (int idx = 0; idx < edu_params.tag_table.E.val.size; idx++)
			edu_params.tag_table.E.val.v[idx] = d(gen);
		for (int idx = 0; idx < etype_table.E.val.size; idx++)
			etype_table.E.val.v[idx] = d(gen);
		return true;
	}

	inline void exportModelParams(ModelUpdate &ada){
		edu_params.exportAdaParams(ada);
		syn_params.exportAdaParams(ada);
		etype_table.exportAdaParams(ada);
		edu_lstm_left_layer1_params.exportAdaParams(ada);
		edu_lstm_right_layer1_params.exportAdaParams(ada);
		scored_action_table.exportAdaParams(ada);
	}

	inline void exportModelBeamParams(ModelUpdate &ada){
		edu_params.exportAdaParams(ada);
		syn_params.exportAdaParams(ada);
		etype_table.exportAdaParams(ada);
		edu_lstm_left_layer1_params.exportAdaParams(ada);
		edu_lstm_right_layer1_params.exportAdaParams(ada);
		scored_action_table.exportAdaParams(ada);
	}

	void saveModel(std::ofstream &os) {
		edu_params.save(os);
		syn_params.save(os);
		etype_table.save(os);
		edu_lstm_left_layer1_params.save(os);
		edu_lstm_right_layer1_params.save(os);
		scored_action_table.save(os);
	}

	void loadModel(std::ifstream &is) {
		edu_params.load(is);
		syn_params.load(is);
		etype_table.load(is, etype_table.elems);
		edu_lstm_left_layer1_params.load(is);
		edu_lstm_right_layer1_params.load(is);
		scored_action_table.load(is, scored_action_table.elems);
	}

};

#endif
