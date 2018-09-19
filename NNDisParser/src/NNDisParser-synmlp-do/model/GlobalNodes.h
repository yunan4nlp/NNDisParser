#ifndef GlobalNodes_H_
#define GlobalNodes_H_

#include "ModelParams.h"
#include "EDUNodes.h"
#include "SynNodes.h"


struct GlobalNodes {
	SynNodes _syn_nodes;
	vector<EDUNodes> _edu_nodes;
	vector<LookupNode> _edu_types;
	vector<ConcatNode> _edu_represents;
	LSTM1Builder _edus_lstm_left_layer1;
	LSTM1Builder _edus_lstm_right_layer1;
	vector<ConcatNode> _edus_lstm_concats;

	inline void resize(const int &max_edu_size, const int &max_sent_size){
		_syn_nodes.resize(max_edu_size * max_sent_size);
		_edu_nodes.resize(max_edu_size);
		for(int i = 0; i < max_edu_size; i++) {
			_edu_nodes[i].resize(max_sent_size);
		}
		_edu_types.resize(max_edu_size);
		_edu_represents.resize(max_edu_size);
		_edus_lstm_left_layer1.resize(max_edu_size);
		_edus_lstm_right_layer1.resize(max_edu_size);
		_edus_lstm_concats.resize(max_edu_size);
	}

	inline void initial(ModelParams &params, const HyperParams &opts){
		resize(opts.maxEDUSize, opts.maxSentSize);
		_syn_nodes.init(&params.syn_params, opts);
		int max_edu_size = _edu_nodes.size();
		for (int i = 0; i < max_edu_size; i++) {
			_edu_nodes[i].initial(&params.edu_params, opts);
			_edu_types[i].setParam(&params.etype_table);
			_edu_types[i].init(opts.etypeDim, opts.dropProb);
			_edu_represents[i].init(opts.eduConcatDim, -1);
		}
		_edus_lstm_left_layer1.init(&params.edu_lstm_left_layer1_params, opts.dropProb, true);
		_edus_lstm_right_layer1.init(&params.edu_lstm_right_layer1_params, opts.dropProb, false);

		for (int i = 0; i < max_edu_size; i++) {
			_edus_lstm_concats[i].init(opts.encoderDim, -1);
		}
	}

	inline void forward(Graph *cg, const Instance &inst) {
		_syn_nodes.forward(cg, inst);
		int edu_size = inst.edus.size();
		int max_size = _edu_nodes.size();
		if (edu_size > max_size)
			edu_size = max_size;
		vector<PNode> edu_hiddens;
		for (int idx = 0; idx < edu_size; idx++) {
			const EDU &cur_edu = inst.edus[idx];
			_edu_nodes[idx].forward(cg, cur_edu, _syn_nodes.outputs[idx]);
			_edu_types[idx].forward(cg, cur_edu.etype);
			_edu_represents[idx].forward(cg, &_edu_nodes[idx].hidden, &_edu_types[idx]);
			//edu_hiddens.push_back(&_edu_nodes[idx].hidden);
			edu_hiddens.push_back(&_edu_represents[idx]);
		}
		_edus_lstm_left_layer1.forward(cg, edu_hiddens);
		_edus_lstm_right_layer1.forward(cg, edu_hiddens);

		for (int idx = 0; idx < edu_size; idx++) {
			_edus_lstm_concats[idx].forward(cg,
				&_edus_lstm_left_layer1._hiddens[idx], &_edus_lstm_right_layer1._hiddens[idx]);
		}
	}
};

#endif