#ifndef GreedyGraph_H_
#define GreedyGraph_H_

#include "ModelParams.h"
#include "GlobalNodes.h"
#include "Action.h"
#include "State.h"
#include "Explorer.h"

class GreedyGraphBuilder {
private:
	ModelParams *pModel;
	HyperParams *pOpts;
public:
	CState start;
	vector<CState> states;
	Explorer explorer;

	vector<vector<COutput> > outputs;
public:
	GlobalNodes globalnodes;

	inline void initial(ModelParams &model, HyperParams &opts) {
		globalnodes.initial(model, opts);
		explorer.initial(opts);
		start.clear();
		start.initial(model, opts);
		states.resize(opts.maxStateSize);
		int statesize = states.size();
		for(int idx = 0; idx < statesize; idx++) {
			states[idx].initial(model, opts);
		}
		pOpts = &opts;
		pModel = &model;
	}

	inline void encode(Graph *cg, const Instance &inst) {
		globalnodes.forward(cg, inst);
	}

	inline void decode(Graph *cg, const Instance &inst, const vector<CAction> *gold_actions = NULL) {
		clearVec(outputs); // clear last outputs
		CState *pGenerator; // the state after best action
		int step = 0; // recored every step
		int offset;
		vector<CAction> candidate_actions; // candidate actions of every candidate actions
		COutput output;
		vector<COutput> per_step_outputs; // store every step outputs
		pGenerator = &start; //  start state and ready to get candidate actions
		start.ready(&inst); // start state get instance data
		CScoredAction scored_action;
		NRHeap<CScoredAction, CScoredActionCompare> beam; // sort the every action score of the state
		beam.resize(pOpts->actionNum + 1); // max action num
		int candidate_action_num; // how many candidate actions of current state
		CAction gold_answer; // the state will move gold action when trainning
		CAction optimal_answer;
		bool gold_action_scored;
		while (true) {
			pGenerator->prepare(globalnodes, *pOpts); // prepare atom feature
			pGenerator->getCandidateActions(candidate_actions, *pOpts); // get the candidate actions
			if (cg->train) {
				gold_answer.set((*gold_actions)[step]); // get gold action when trainning
				explorer.getOracle(*pGenerator, inst.result.subtrees, optimal_answer);
			}
			pGenerator->computeNextActionScore(cg, candidate_actions, true); // calculate the scores of candidate actions
			cg->compute(); // compute score of candidate actions
			beam.clear();
			per_step_outputs.clear();
			scored_action.state = pGenerator; // get current state
			candidate_action_num = candidate_actions.size();
			gold_action_scored = false;
			for (int idx = 0; idx < candidate_action_num; idx++) {
				scored_action.ac.set(candidate_actions[idx]); // get current action

				// is gold state and move to gold action?
				if (scored_action.ac == optimal_answer) {
					scored_action.is_gold = true;
					gold_action_scored = true;
					output.is_gold = true;
				} else {
					scored_action.is_gold = false;
					output.is_gold = false;
				}

				if (cg->train && scored_action.ac != optimal_answer) pGenerator->_next_action_score.outputs[idx].val[0] += pOpts->delta;
				scored_action.position = idx; // recored the position.
				scored_action.score = pGenerator->_next_action_score.outputs[idx].val[0]; // get current action score
				output.in = &pGenerator->_next_action_score.outputs[idx];
				beam.add_elem(scored_action);
				per_step_outputs.push_back(output);
			}
			outputs.push_back(per_step_outputs);

			if (cg->train && !gold_action_scored) {
				cout << "gold action don't scored....";
				cout << ", step:" << step;
				cout << ", action: " << optimal_answer.str(*pOpts) << endl;
			}
			offset = beam.elemsize();
			if (offset == 0) {
				cout << "error, beam size = 0" << endl;
			}

			beam.sort_elem(); // sort score of actions, beam[0] will be highest score.
			const CScoredAction& highest_score_action = beam[0];
			if (cg->train) {
				bool find_next = false;
				for (int idx = 0; idx < offset; idx++) {
					const CScoredAction& current_scored_action = beam[idx];
					// if trainning model, move the gold action
					if (current_scored_action.is_gold) {
						if(!pOpts->dynamicOracle)
							current_scored_action.state->move(&states[step], current_scored_action.ac);
						else {
							dtype p = rand() / double(RAND_MAX);
							if (p < pOpts->oracleProb)
								current_scored_action.state->move(&states[step], current_scored_action.ac);
							else
								highest_score_action.state->move(&states[step], highest_score_action.ac); // move the highest score
						}
						find_next = true;
						break;
					}
				}
				if(!find_next) {
					cout << "error... can' t find next" << endl;
				}
			} else {
				// if not trainning model, move to highest score action
				highest_score_action.state->move(&states[step], highest_score_action.ac); // move the highest score
			}
			pGenerator = &states[step]; // pGenerator is the state after moving to the highest score.
			if (states[step].isEnd()) // finish
				break;
			step++;
		}

	}
};

#endif /* GreedyGraph_H_ */
