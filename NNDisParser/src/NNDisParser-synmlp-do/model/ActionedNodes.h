#ifndef ACTIONED_NODES_H 
#define ACTIONED_NODES_H

#include "ModelParams.h"
#include "AtomFeature.h"

// score the action one by one
class ActionedNodes {
public:
	BucketNode edu_bucket;
	AvgPoolNode subtree_stack0;
	AvgPoolNode subtree_stack1;
	AvgPoolNode subtree_stack2;

	ConcatNode state_concat;

	vector<ActionNode> current_action_inputs; // weight of every action
	vector<PAddNode> outputs;


	const HyperParams *pOpts;

	inline void initial(ModelParams &params, const HyperParams &hyparams) {
		current_action_inputs.resize(hyparams.actionNum);
		outputs.resize(hyparams.actionNum);

		edu_bucket.init(hyparams.encoderDim, -1);
		subtree_stack0.init(hyparams.encoderDim, -1);
		subtree_stack1.init(hyparams.encoderDim, -1);
		subtree_stack2.init(hyparams.encoderDim, -1);

		for(int idx = 0; idx < hyparams.actionNum; idx++) {
			current_action_inputs[idx].setParam(&params.scored_action_table);
			current_action_inputs[idx].init(1, -1);
			outputs[idx].init(1, -1);
		}

		state_concat.init(hyparams.stateConcatDim, -1);
		
		pOpts = &hyparams;
	}

	inline void forward(Graph *cg, const vector<CAction> &actions,  AtomFeat &atomFeat, PNode prevStateNode) {
		vector<PNode> state_feats;
		edu_bucket.forward(cg, 0);
		PNode pedu_lstm_buffer0 =
			atomFeat._next_index >= 0 ? (PNode) &(*atomFeat._pedu_lstm)[atomFeat._next_index] : (PNode)&edu_bucket;

		PNode pedu_lstm_stack0;
		int edu_start, edu_end;
		if (atomFeat._stack_top_0 != NULL) {
			edu_start = atomFeat._stack_top_0->edu_start;
			edu_end = atomFeat._stack_top_0->edu_end;
			if (edu_start == edu_end)
				pedu_lstm_stack0 = (PNode)&(*atomFeat._pedu_lstm)[edu_start];
			else {
				vector<PNode> edu_nodes;
				for(int idx = edu_start; idx <= edu_end; idx++){
					edu_nodes.push_back(&(*atomFeat._pedu_lstm)[idx]);
				}
				subtree_stack0.forward(cg, edu_nodes);
				pedu_lstm_stack0 = &subtree_stack0;
			}
		}
		else
			pedu_lstm_stack0 = &edu_bucket;

		PNode pedu_lstm_stack1;
		if (atomFeat._stack_top_1 != NULL) {
			edu_start = atomFeat._stack_top_1->edu_start;
			edu_end = atomFeat._stack_top_1->edu_end;
			if (edu_start == edu_end)
				pedu_lstm_stack1 = (PNode)&(*atomFeat._pedu_lstm)[edu_start];
			else {
				vector<PNode> edu_nodes;
				for(int idx = edu_start; idx <= edu_end; idx++){
					edu_nodes.push_back(&(*atomFeat._pedu_lstm)[idx]);
				}
				subtree_stack1.forward(cg, edu_nodes);
				pedu_lstm_stack1 = &subtree_stack1;
			}
		}
		else
			pedu_lstm_stack1 = &edu_bucket;

		PNode pedu_lstm_stack2;
		if (atomFeat._stack_top_2 != NULL) {
			edu_start = atomFeat._stack_top_2->edu_start;
			edu_end = atomFeat._stack_top_2->edu_end;
			if (edu_start == edu_end)
				pedu_lstm_stack2 = (PNode)&(*atomFeat._pedu_lstm)[edu_start];
			else {
				vector<PNode> edu_nodes;
				for(int idx = edu_start; idx <= edu_end; idx++){
					edu_nodes.push_back(&(*atomFeat._pedu_lstm)[idx]);
				}
				subtree_stack2.forward(cg, edu_nodes);
				pedu_lstm_stack2 = &subtree_stack2;
			}
		}
		else
			pedu_lstm_stack2 = &edu_bucket;

		state_feats.push_back(pedu_lstm_buffer0);
		state_feats.push_back(pedu_lstm_stack0);
		state_feats.push_back(pedu_lstm_stack1);
		state_feats.push_back(pedu_lstm_stack2);

		state_concat.forward(cg, state_feats);
		int cur_action_num = actions.size();
		vector<PNode> sumNodes;
		for (int idx = 0; idx < cur_action_num; idx++) {
			sumNodes.clear();
			const string &action_str = actions[idx].str(*pOpts);
			current_action_inputs[idx].forward(cg, action_str, &state_concat);

			if (prevStateNode != NULL)
				sumNodes.push_back(prevStateNode);

			sumNodes.push_back(&current_action_inputs[idx]);
			outputs[idx].forward(cg, sumNodes);
		}
	}
	
};

#endif /*ACTIONED_NODES_H*/