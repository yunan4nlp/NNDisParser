#ifndef BeamGraph_H_
#define BeamGraph_H_

#include "ModelParams.h"
#include "GlobalNodes.h"
#include "Action.h"
#include "State.h"

// beam search
class BeamGraphBuilder {
private:
	ModelParams *pModel;
	HyperParams *pOpts;
public:
	CState start;
	vector<vector<CState> > states;
	vector<vector<COutput> > outputs;
public:
	GlobalNodes globalnodes;

	inline void initial(ModelParams &model, HyperParams &opts) {
		globalnodes.initial(model, opts);
		start.clear();
		start.initial(model, opts);
		states.resize(opts.maxStateSize);
		for (int idx = 0; idx < opts.maxStateSize; idx++) {
			states[idx].resize(opts.beam);
			for (int idy = 0; idy < opts.beam; idy++) {
				states[idx][idy].initial(model, opts);
			}
		}
		pOpts = &opts;
		pModel = &model;
	}

	inline void encode(Graph *cg, const Instance &inst) {
		globalnodes.forward(cg, inst);
	}

	inline void decode(Graph *cg, const Instance &inst, const vector<CAction> *gold_actions = NULL) {
		//first step, clear node values
		clearVec(outputs);

		//second step, build graph
		vector<CState* > lastStates;
		CState* pGenerator;
		int step, offset;
		vector<vector<CAction> > actions; // actions to apply for a candidate
		CScoredAction scored_action; // used rank actions
		COutput output;
		bool correct_action_scored;
		bool correct_in_beam;
		CAction answer, action;
		vector<COutput> per_step_output;
		NRHeap<CScoredAction, CScoredActionCompare> beam;
		beam.resize(pOpts->beam);
		actions.resize(pOpts->beam);

		lastStates.clear();
		start.ready(&inst);
		lastStates.push_back(&start);

		step = 0;
		while (true) {
			//prepare for the next
			for (int idx = 0; idx < lastStates.size(); idx++) {
				pGenerator = lastStates[idx];
				pGenerator->prepare(globalnodes, *pOpts);
			}

			for (int idx = 0; idx < lastStates.size(); idx++) {
				pGenerator = lastStates[idx];
				pGenerator->getCandidateActions(actions[idx], *pOpts);
				pGenerator->computeNextActionScore(cg, actions[idx], false);
			}

			cg->compute(); //must compute here, or we can not obtain the scores

			answer.clear();
			per_step_output.clear();
			correct_action_scored = false;
			if (cg->train) answer = (*gold_actions)[step];
			beam.clear();

			for (int idx = 0; idx < lastStates.size(); idx++) {
				pGenerator = lastStates[idx];
				scored_action.state = pGenerator;
				for (int idy = 0; idy < actions[idx].size(); ++idy) {
					scored_action.ac.set(actions[idx][idy]); //TODO:
					if (pGenerator->_is_gold && actions[idx][idy] == answer) {
						scored_action.is_gold = true;
						correct_action_scored = true;
						output.is_gold = true;
					}
					else {
						scored_action.is_gold = false;
						output.is_gold = false;
					}
					if (cg->train && actions[idx][idy] != answer) pGenerator->_next_action_score.outputs[idx].val[0] += pOpts->delta;
					scored_action.score = pGenerator->_next_action_score.outputs[idy].val[0];
					scored_action.position = idy;
					output.in = &(pGenerator->_next_action_score.outputs[idy]);
					beam.add_elem(scored_action);
					per_step_output.push_back(output);
				}
			}

			outputs.push_back(per_step_output);

			if (cg->train && !correct_action_scored) { //training
				std::cout << "error during training, gold-standard action is filtered: " << step << std::endl;
				std::cout << answer.str(*pOpts) << std::endl;
				for (int idx = 0; idx < lastStates.size(); idx++) {
					pGenerator = lastStates[idx];
					if (pGenerator->_is_gold) {
						pGenerator->getCandidateActions(actions[idx], *pOpts);
						for (int idy = 0; idy < actions[idx].size(); ++idy) {
							std::cout << actions[idx][idy].str(*pOpts) << " ";
						}
						std::cout << std::endl;
					}
				}
				return;
			}

			offset = beam.elemsize();
			if (offset == 0) { // judge correctiveness
				std::cout << "error, reach no output here, please find why" << std::endl;
				/*
				for (int idx = 0; idx < pCharacters->size(); idx++) {
					std::cout << (*pCharacters)[idx] << std::endl;
				}
				std::cout << "" << std::endl;
				*/
				return;
			}

			beam.sort_elem();
			for (int idx = 0; idx < offset; idx++) {
				pGenerator = beam[idx].state;
				action.set(beam[idx].ac);
				pGenerator->move(&(states[step][idx]), action);
				states[step][idx]._is_gold = beam[idx].is_gold;
				states[step][idx]._score_node = &(pGenerator->_next_action_score.outputs[beam[idx].position]);
			}

			if (states[step][0].isEnd()) {
				break;
			}

			//for next step
			lastStates.clear();
			correct_in_beam = false;
			for (int idx = 0; idx < offset; idx++) {
				lastStates.push_back(&(states[step][idx]));
				if (lastStates[idx]->_is_gold) {
					correct_in_beam = true;
				}
			}

			if (cg->train && !correct_in_beam) {
				break;
			}

			step++;
		}

		return;
	}
};

#endif /* BeamGraph_H_ */