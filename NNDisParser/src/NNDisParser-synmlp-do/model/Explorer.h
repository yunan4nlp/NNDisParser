#ifndef Explorer_H_
#define Explorer_H_


class Explorer {

public:
	HyperParams *pOpts;
	inline void initial(HyperParams &opts) {
		pOpts = &opts;
	}

private:

	inline int nuclearLabelLoss(const CAction &ac, const CState &error_state, const vector<SubTree> &gold_tree) {
		assert(error_state._stack_size >= 2);
		const CNode &top0 = error_state._stack[error_state._stack_size - 1];
		const CNode &top1 = error_state._stack[error_state._stack_size - 2];
		SubTree subtree0, subtree1;
		if (ac._nuclear == CAction::NN) {
			subtree0.edu_start = top0.edu_start;
			subtree0.edu_end = top0.edu_end;
			subtree0.nuclear = NUCLEAR;
			subtree0.relation = pOpts->labelAlpha.from_id(ac._label);
			subtree1.edu_start = top1.edu_start;
			subtree1.edu_end = top1.edu_end;
			subtree1.nuclear = NUCLEAR;
			subtree1.relation = pOpts->labelAlpha.from_id(ac._label);
		}
		else if (ac._nuclear == CAction::NS) {
			subtree0.edu_start = top0.edu_start;
			subtree0.edu_end = top0.edu_end;
			subtree0.nuclear = SATELLITE;
			subtree0.relation = pOpts->labelAlpha.from_id(ac._label);
			subtree1.edu_start = top1.edu_start;
			subtree1.edu_end = top1.edu_end;
			subtree1.nuclear = NUCLEAR;
			subtree1.relation = SPAN;
		}
		else if (ac._nuclear == CAction::SN) {
			subtree0.edu_start = top0.edu_start;
			subtree0.edu_end = top0.edu_end;
			subtree0.nuclear = NUCLEAR;
			subtree0.relation = SPAN;
			subtree1.edu_start = top1.edu_start;
			subtree1.edu_end = top1.edu_end;
			subtree1.nuclear = SATELLITE;
			subtree1.relation = pOpts->labelAlpha.from_id(ac._label);
		}

		int loss0 = subtreeLoss(subtree0, gold_tree);
		int loss1 = subtreeLoss(subtree1, gold_tree);
		return loss0 + loss1;
	}

	inline int subtreeLoss(const SubTree &subtree, const vector<SubTree> &gold_tree) {
		int subtree_size = gold_tree.size(), loss = 3;
		for (int idx = 0; idx < subtree_size; idx++) {
			const SubTree &gold_subtree = gold_tree[idx];
			if (subtree.spanEqual(gold_subtree)) {
				loss--;
				if (subtree.nuclear == gold_subtree.nuclear) {
					loss--;
					if (subtree.relation == gold_subtree.relation)
						loss--;
				}
				break;
			}
		}
		return loss;
	}

	inline int shiftLoss(const CState &error_state, const vector<SubTree> &gold_tree) {
		assert(error_state._stack_size >= 1);
		int start, end = error_state._stack[error_state._stack_size - 1].edu_end;
		int gold_action_size = gold_tree.size(), count = 0, max_size = error_state._stack_size - 1;
		for (int idx = 0; idx < max_size; idx++) {
			start = error_state._stack[idx].edu_start;
			for (int idy = 0; idy < gold_action_size; idy++) {
				const SubTree &gold_subtree = gold_tree[idy];
				if (start == gold_subtree.edu_start && end == gold_subtree.edu_end) {
					count++;
				}
			}
		}
		return count;
	}

	inline int reduceLoss(const CState &error_state, const vector<SubTree> &gold_tree) {
		assert(error_state._stack_size >= 1);
		int start = error_state._stack[error_state._stack_size - 1].edu_start, end;
		int gold_action_size = gold_tree.size(), count = 0;
		for (int idx = error_state._next_index; idx < error_state._edu_size; idx++) {
			end = idx;
			for (int idy = 0; idy < gold_action_size; idy++) {
				const SubTree &gold_subtree = gold_tree[idy];
				if (start == gold_subtree.edu_start && end == gold_subtree.edu_end)
					count++;
			}
		}
		return count;
	}

	inline void getReduceCandidate(const CState &error_state, const vector<SubTree> &gold_tree, vector<CAction> &candidate_actions) {
		assert(error_state._stack_size >= 2);
		int label_size = pOpts->labelAlpha.size();
		string action_str;
		CAction ac;
		vector<pair<CAction, int> > tmp_acts;
		for (int idx = 0; idx < 3; idx++) {
			for (int idy = 0; idy < label_size; idy++) {
				ac.set(CAction::REDUCE, idx, idy);
				action_str = ac.str(*pOpts);
				if (pOpts->actionAlpha.from_string(action_str) >= 0) {
					int loss = nuclearLabelLoss(ac, error_state, gold_tree);
					tmp_acts.push_back(make_pair(ac, loss));
					if (loss == 0) {
						candidate_actions.push_back(ac);
						return;
					}
				}
			}
		}
		assert(tmp_acts.size() > 0);
		int action_size = tmp_acts.size(), min_loss = tmp_acts[0].second, min_index = 0, cur_loss;
		for (int idx = 1; idx < action_size; idx++) {
			auto &cur_iter = tmp_acts[idx];
			cur_loss = cur_iter.second;
			if (cur_loss < min_loss) {
				min_index = idx;
				min_loss = cur_loss;
			}
		}

		for (int idx = 0; idx < action_size; idx++) {
			auto &cur_iter = tmp_acts[idx];
			if (cur_iter.second == min_loss)
				candidate_actions.push_back(cur_iter.first);
		}
	}

public:
	inline void getOracle(const CState &error_state, const vector<SubTree> &gold_tree, CAction &optimal_action) {
		vector<CAction> candidate_actions;
		CAction ac;
		if (error_state._stack_size < 2) {
			if (error_state._next_index == error_state._edu_size)
				ac.set(CAction::POP_ROOT, -1, -1);
			else
				ac.set(CAction::SHIFT, -1, -1);
			candidate_actions.push_back(ac);
		}
		else if (error_state._next_index == error_state._edu_size) {
			ac.set(CAction::REDUCE, -1, -1);
		}
		else {
			int shift_loss = shiftLoss(error_state, gold_tree);
			int reduce_loss = reduceLoss(error_state, gold_tree);
			if (shift_loss < reduce_loss) {
				ac.set(CAction::SHIFT, -1, -1);
				candidate_actions.push_back(ac);
			}
			if (shift_loss >= reduce_loss) {
				ac.set(CAction::REDUCE, -1, -1);
				if (shift_loss == reduce_loss) { 
					CAction shift_action(CAction::SHIFT, -1, -1);
					candidate_actions.push_back(shift_action);
				}
			}
		}
		if (ac.isReduce()) {
			getReduceCandidate(error_state, gold_tree, candidate_actions);
		}
		int min = 0, max = candidate_actions.size();
		assert(max > 0);
		int rand_index = min + (rand() * (double)(max - min) / RAND_MAX);
		optimal_action.set(candidate_actions[rand_index]);
	}
};

#endif