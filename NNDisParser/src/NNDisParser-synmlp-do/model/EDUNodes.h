#ifndef EDUNODES
#define EDUNODES

struct EDUParams {
	LookupTable word_table;
	LookupTable tag_table;

	UniParams word_represent_params;
	LSTM1Params word_lstm_left_params;
	LSTM1Params word_lstm_right_params;

	LSTM1Params syn_lstm_left_params;
	LSTM1Params syn_lstm_right_params;

	inline void initial(const HyperParams& opts){
		word_represent_params.initial(opts.wordRepresentHiddenSize, opts.wordConcatDim, true);
		word_lstm_left_params.initial(opts.rnnHiddenSize, opts.hiddenSize);
		word_lstm_right_params.initial(opts.rnnHiddenSize, opts.hiddenSize);

		syn_lstm_left_params.initial(opts.rnnHiddenSize, opts.hiddenSize);
		syn_lstm_right_params.initial(opts.rnnHiddenSize, opts.hiddenSize);
	}

	inline void exportAdaParams(ModelUpdate& ada){
		word_table.exportAdaParams(ada);
		tag_table.exportAdaParams(ada);
		word_represent_params.exportAdaParams(ada);
		word_lstm_left_params.exportAdaParams(ada);
		word_lstm_right_params.exportAdaParams(ada);

		syn_lstm_left_params.exportAdaParams(ada);
		syn_lstm_right_params.exportAdaParams(ada);
	}
};

class EDUNodes{
public:
	vector<LookupNode> word_inputs;
	vector<LookupNode> tag_inputs;
	vector<ConcatNode> word_concat;
	vector<UniNode> word_represents;
	LSTM1Builder word_lstm_left;
	LSTM1Builder word_lstm_right;
	vector<ConcatNode> word_lstm_concat;
	AvgPoolNode word_avg_pooling;

	LSTM1Builder syn_lstm_left;
	LSTM1Builder syn_lstm_right;
	vector<ConcatNode> syn_lstm_concat;
	AvgPoolNode syn_avg_pooling;

	ConcatNode hidden;

	const HyperParams *pOpts;


	void resize(const int &word_size){
		word_inputs.resize(word_size);
		tag_inputs.resize(word_size);
		word_concat.resize(word_size);
		word_represents.resize(word_size);
		word_lstm_left.resize(word_size);
		word_lstm_right.resize(word_size);
		word_lstm_concat.resize(word_size);

		syn_lstm_left.resize(word_size);
		syn_lstm_right.resize(word_size);
		syn_lstm_concat.resize(word_size);
	}

	inline void initial(EDUParams *params, const HyperParams &opts){
		pOpts = &opts;
		int maxsize = word_inputs.size();
		for (int idx = 0; idx < maxsize; idx++) {
			word_inputs[idx].setParam(&params->word_table);
			word_inputs[idx].init(opts.wordDim, opts.dropProb);
			tag_inputs[idx].setParam(&params->tag_table);
			tag_inputs[idx].init(opts.tagDim, opts.dropProb);
			word_concat[idx].init(opts.wordConcatDim, -1);
			word_represents[idx].setParam(&params->word_represent_params);
			word_represents[idx].init(opts.wordRepresentHiddenSize, -1);
		}
		word_lstm_left.init(&params->word_lstm_left_params, opts.dropProb, true);
		word_lstm_right.init(&params->word_lstm_right_params, opts.dropProb, false);
		for (int idx = 0; idx < maxsize; idx++) {
			word_lstm_concat[idx].init(opts.rnnHiddenSize * 2, -1);
		}
		word_avg_pooling.init(opts.rnnHiddenSize * 2, -1);

		syn_lstm_left.init(&params->syn_lstm_left_params, opts.dropProb, true);
		syn_lstm_right.init(&params->syn_lstm_right_params, opts.dropProb, false);
		for (int idx = 0; idx < maxsize; idx++)
			syn_lstm_concat[idx].init(opts.rnnHiddenSize * 2, -1);
		syn_avg_pooling.init(opts.rnnHiddenSize * 2, -1);

		hidden.init(opts.eduHiddenDim, -1);
	}

	inline void forward(Graph *cg, const EDU &edu, const vector<PNode> &syn_outputs) {
		int word_size = edu.words.size(), max_size = word_inputs.size();
		if (word_size > max_size)
			word_size = max_size;
		string cur_word;
		for (int idx = 0; idx < word_size; idx++) {
			cur_word = edu.words[idx];
			// Unknown word strategy: STOCHASTIC REPLACEMENT
			auto it = pOpts->word_stat.find(cur_word);
			int c;
			if (it == pOpts->word_stat.end())
				c = 0;
			else
				c = it->second;
			dtype rand_drop = rand() / double(RAND_MAX);
			if (cg->train && c <= 1 && rand_drop < 0.5) {
				cur_word = unknownkey;
			}

			const string& cur_tag = edu.tags[idx];
			word_inputs[idx].forward(cg, cur_word);
			tag_inputs[idx].forward(cg, cur_tag);
			word_concat[idx].forward(cg, &word_inputs[idx], &tag_inputs[idx]);
		}

		for (int idx = 0; idx < word_size; idx++) {
			assert(word_concat[idx].dim == word_represents[idx].param->W.inDim());
			word_represents[idx].forward(cg, &word_concat[idx]);
		}
		word_lstm_left.forward(cg, getPNodes(word_represents, word_size));
		word_lstm_right.forward(cg, getPNodes(word_represents, word_size));

		for (int idx = 0; idx < word_size; idx++) {
			word_lstm_concat[idx].forward(cg, &word_lstm_left._hiddens[idx], &word_lstm_right._hiddens[idx]);
		}
		word_avg_pooling.forward(cg, getPNodes(word_lstm_concat, word_size));

		syn_lstm_left.forward(cg, syn_outputs);
		syn_lstm_right.forward(cg, syn_outputs);

		for (int idx = 0; idx < word_size; idx++) {
			syn_lstm_concat[idx].forward(cg, &syn_lstm_left._hiddens[idx], &syn_lstm_right._hiddens[idx]);
		}
		syn_avg_pooling.forward(cg, getPNodes(syn_lstm_concat, word_size));

		hidden.forward(cg, &word_avg_pooling,
			&syn_avg_pooling);
	}
};

#endif