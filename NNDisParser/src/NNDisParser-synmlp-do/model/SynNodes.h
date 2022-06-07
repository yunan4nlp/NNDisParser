#ifndef SynNodes_H_
#define SynNodes_H_

struct SynParams {
	UniParams syn_combine;

	inline void initial(const HyperParams& opts) {
		syn_combine.initial(opts.hiddenSize, opts.synCombineDim, true);
	}

	inline void exportAdaParams(ModelUpdate& ada) {
		syn_combine.exportAdaParams(ada);
	}

	inline void save(std::ofstream &os){
		syn_combine.save(os);
	}

	inline void load(std::ifstream &is){
		syn_combine.load(is);
	}
};

struct SynNodes{
	vector<ConcatNode> syn_concat;
	vector<UniNode> hiddens;
	vector<vector<PNode> > outputs;
	void resize(const int &word_size) {
		syn_concat.resize(word_size);
		hiddens.resize(word_size);
	}

	const HyperParams *pOpts;

	inline void init(SynParams *params, const HyperParams &opts) { 
		pOpts = &opts;
		int maxsize = syn_concat.size();
		for (int idx = 0; idx < maxsize; idx++) {
			syn_concat[idx].init(opts.synCombineDim, -1);
			hiddens[idx].setParam(&params->syn_combine);
			hiddens[idx].init(opts.hiddenSize, opts.dropProb);
		}
	}

	void forward(Graph *cg, const Instance &inst){
		assert(inst.syn_feats.size() == inst.words.size());
		int syn_size = inst.syn_feats.size();
		int max_size = syn_concat.size();
		if (syn_size > max_size)
			syn_size = max_size;

		for (int idx = 0; idx < syn_size; idx++) {
			inst.syn_feats[idx][0]->forward(cg);
			inst.syn_feats[idx][1]->forward(cg);
			inst.syn_feats[idx][2]->forward(cg);
			inst.syn_feats[idx][3]->forward(cg);
		}

		for (int idx = 0; idx < syn_size; idx++) {
			syn_concat[idx].forward(cg,
				inst.syn_feats[idx][0], inst.syn_feats[idx][1],
				inst.syn_feats[idx][2], inst.syn_feats[idx][3]);
		}

		for (int idx = 0; idx < syn_size; idx++) {
			hiddens[idx].forward(cg, &syn_concat[idx]);
		}

		int word_size;
		int edu_size = inst.edus.size();
		if (edu_size > pOpts->maxEDUSize)
			edu_size = pOpts->maxEDUSize;
		outputs.clear();
		outputs.resize(edu_size);
		int offset = 0;
		for (int idx = 0; idx < edu_size; idx++) {
			word_size = inst.edus[idx].words.size();
			if (word_size > pOpts->maxSentSize)
				word_size = pOpts->maxSentSize;
			vector<PNode> &cur_output = outputs[idx];
			for (int idy = 0; idy < word_size; idy++)
			{
				assert(offset < syn_size);
				cur_output.push_back(&hiddens[offset]);
				offset++;
			}
		}
	}
};

#endif