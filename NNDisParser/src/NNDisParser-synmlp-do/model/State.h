#ifndef CState_H_
#define CState_H_

#include "ActionedNodes.h"
#include "AtomFeature.h"


class CState {
public:

	CNode _stack[max_length];
	short _stack_size;
	int _edu_size;
	const Instance *inst;
	short _next_index; // index of top element of buffer
	CState *_pre_state;
	CAction _pre_action;
	bool _is_start;
	ActionedNodes _next_action_score;
	AtomFeat _atom_feat;
	PNode _score_node; // the score node of this state
	bool _is_gold;

public:
	inline void initial(ModelParams &params, const HyperParams &hyparams) {
		_next_action_score.initial(params, hyparams);
	}

public:
	CState() {
		clear();
	}

	~CState() {
	}

public:
	void clear() {
		_next_index = 0;
		_stack_size = 0;
		inst = NULL;
		_pre_state = NULL;
		_score_node = NULL;
		_is_start = true;
		_is_gold = true;
	}

	void computeNextActionScore(Graph *cg, const vector<CAction> &candidate_actions, bool is_greedy) {
		if (_is_start || is_greedy)
			_next_action_score.forward(cg, candidate_actions, _atom_feat, NULL);
		else
			_next_action_score.forward(cg, candidate_actions, _atom_feat, _score_node);
	}

	void getResults(CResult &result, const HyperParams& opts) const {
		result.clear();
		const CState *state_iter = this;
		while (!state_iter->_pre_state->_is_start) {
			const CAction &action = state_iter->_pre_action;
			const CState *pre_state = state_iter->_pre_state;
			if (action.isReduce()) {
				assert(pre_state->_stack_size >= 2);
				const CNode &right_node = pre_state->_stack[pre_state->_stack_size - 1];
				const CNode &left_node = pre_state->_stack[pre_state->_stack_size - 2];
				SubTree left_subtree, right_subtree;

				left_subtree.edu_start = left_node.edu_start;
				left_subtree.edu_end = left_node.edu_end;

				right_subtree.edu_start = right_node.edu_start;
				right_subtree.edu_end = right_node.edu_end;
				
				if(action._nuclear == CAction::NN) {
					left_subtree.nuclear = NUCLEAR;
					right_subtree.nuclear = NUCLEAR;
					left_subtree.relation = opts.labelAlpha.from_id(action._label);
					right_subtree.relation = opts.labelAlpha.from_id(action._label);
				}
				else if (action._nuclear == CAction::SN) {
					left_subtree.nuclear = SATELLITE;
					right_subtree.nuclear = NUCLEAR;
					left_subtree.relation = opts.labelAlpha.from_id(action._label);
					right_subtree.relation = SPAN;
				}
				else if (action._nuclear == CAction::NS) {
					left_subtree.nuclear = NUCLEAR;
					right_subtree.nuclear = SATELLITE;
					left_subtree.relation = SPAN;
					right_subtree.relation = opts.labelAlpha.from_id(action._label);
				}
				result.subtrees.insert(result.subtrees.begin(), right_subtree);
				result.subtrees.insert(result.subtrees.begin(), left_subtree);
			}
			state_iter = state_iter->_pre_state;
		} 
	}

	// prepare instance
	inline void ready(const Instance *pInst) {
		this->inst = pInst;
		_edu_size = pInst->edus.size();
	}

	// copy data to next state	
	void copyState(CState *next) {
		memcpy(next->_stack, _stack, sizeof(CNode) * _stack_size);
		next->_edu_size = _edu_size;
		next->inst = inst;
		next->_pre_state = this;
	}

	// temp mark
	inline void doneMark() {
		_stack[_stack_size].clear();
	}

	// for all states, not for gold state
	bool allowShift() const {
		if (_next_index == _edu_size)
			return false;
		else
			return true;
	}
	// for all states, not for gold state
	bool allowReduce() const {
		if (_stack_size >= 2)
			return true;
		else
			return false;
	}
	// for all states, not for gold state	
	bool allowPopRoot() const {
		if (_next_index == _edu_size && _stack_size == 1)
			return true;
		else
			return false;
	}

	// shift action, top of buffer -> stack
	void shift(CState *next) {
		next->_stack_size = _stack_size + 1;
		next->_next_index = _next_index + 1;
		copyState(next);
		CNode &top = next->_stack[next->_stack_size - 1];
		top.clear();
		top.is_validate = true;
		top.edu_start = _next_index;
		top.edu_end = _next_index;
		next->_pre_action.set(CAction::SHIFT, -1, -1);
		next->doneMark();
	}

	// reduce
	void reduce(CState *next, const short &nuclear, const short &label) {
		next->_stack_size = _stack_size - 1;
		next->_next_index = _next_index;
		copyState(next);
		CNode &top0 = next->_stack[_stack_size - 1];
		CNode &top1 = next->_stack[_stack_size - 2];
		assert(top0.edu_start == top1.edu_end + 1);
		assert(top0.is_validate && top1.is_validate);
		top1.edu_end = top0.edu_end;
		top1.nuclear = nuclear;
		top1.label = label;
		top0.clear();
		next->_pre_action.set(CAction::REDUCE, nuclear, label);
		next->doneMark();
	}

	void popRoot(CState *next) {
		next->_stack_size = 0;
		next->_next_index = _edu_size;
		copyState(next);
		CNode &top0 = next->_stack[_stack_size - 1];
		assert(top0.edu_start == 0 && top0.edu_end + 1 == inst->edus.size());
		assert(top0.is_validate);
		top0.clear();
		next->_pre_action.set(CAction::POP_ROOT, -1, -1);
		next->doneMark();
	}

	bool isEnd() const {
		if (_pre_action.isFinish())
			return true;
		else
			return false;
	}

	//move to next state 
	void move(CState *next, const CAction &ac) {
		next->_is_start = false; // if a state move to next, next state can't be start 
		next->_is_gold = false; // here we don't know the action is gold or not
		if (ac.isShift())
			shift(next);
		else if (ac.isReduce())
			reduce(next, ac._nuclear, ac._label);
		else if (ac.isFinish())
			popRoot(next);
		else {
			std::cerr << "error aciton" << endl;
		}
	}
	// get candidate actions of this state.
	void getCandidateActions(vector<CAction>& actions, HyperParams& opts) const {
		actions.clear();
		CAction ac;

		if (isEnd()) {
			actions.push_back(ac);
			return;
		}

		if (allowShift()) { 
			ac.set(CAction::SHIFT, -1, -1);
			actions.push_back(ac);
		}

		if (allowReduce()) {
			int label_size = opts.labelAlpha.size();
			string action_str;
			for (int idx = 0; idx < 3; idx++) {
				for (int idy = 0; idy < label_size; idy++) {
					ac.set(CAction::REDUCE, idx, idy);
					action_str = ac.str(opts);
					if (opts.actionAlpha.from_string(action_str) >= 0) {
						actions.push_back(ac);
					}
				}
			}
		}

		if (allowPopRoot()) {
			ac.set(CAction::POP_ROOT, -1, -1);
			actions.push_back(ac);
		}

	}
	// prepare atom feature of this state
	void prepare(GlobalNodes& globelnodes, const HyperParams &hyperparms) {
		_atom_feat._pedu_lstm = &globelnodes._edus_lstm_concats;
		_atom_feat._next_index = _next_index >= 0 && _next_index < _edu_size ? _next_index : -1;

		_atom_feat._stack_top_0 = _stack_size > 0 ? &_stack[_stack_size - 1] : NULL;
		_atom_feat._stack_top_1 = _stack_size > 1 ? &_stack[_stack_size - 2] : NULL;
		_atom_feat._stack_top_2 = _stack_size > 2 ? &_stack[_stack_size - 3] : NULL;

/*
		_atom_feat._nuclear_top_0 = _stack_size > 0 ? _stack[_stack_size - 1].nuclear_str() : nullkey;
		_atom_feat._nuclear_top_1 = _stack_size > 1 ? _stack[_stack_size - 2].nuclear_str() : nullkey;
		_atom_feat._nuclear_top_2 = _stack_size > 2 ? _stack[_stack_size - 3].nuclear_str() : nullkey;
		_atom_feat._pre_action_str =
			_is_start ? nullkey : _pre_action.str(hyperparms);
		_atom_feat._pre_pre_action_str =
			_pre_state == NULL || _pre_state->_is_start ? nullkey : _pre_state->_pre_action.str(hyperparms);
		_atom_feat._pre_action_lstm =
			/ _pre_state == NULL ? NULL : &_pre_state->_next_action_score.action_lstm;
		int buffer_size = _edu_size - _next_index;
		_atom_feat._buffer_size = buffer_size > 20 ? 20 : buffer_size;

		_atom_feat._stack_size = _stack_size > 5 ? 5 : _stack_size;
		*/
	}

};


class CScoredAction {
public:
	CState* state; // current state
	dtype score; // the score of action

	CAction ac;
	bool is_gold; // is gold state && move to gold action ? true:false;
	int position; // the position of action.

	CScoredAction() {
		state = NULL;
		score = 0;
		is_gold = false;
		position = -1;
	}
public:
	bool operator <(const CScoredAction &a1) const {
		return score < a1.score;
	}
	bool operator >(const CScoredAction &a1) const {
		return score > a1.score;
	}
	bool operator <=(const CScoredAction &a1) const {
		return score <= a1.score;
	}
	bool operator >=(const CScoredAction &a1) const {
		return score >= a1.score;
	}
};

class CScoredActionCompare {
public:
	int operator()(const CScoredAction &o1, const CScoredAction &o2) const {
		if (o1.score < o2.score)
			return -1;
		else if (o1.score > o2.score)
			return 1;
		else
			return 0;
	}
};

class COutput {
public:
	PNode in;
	bool is_gold;

	COutput() {
		in = NULL;
		is_gold = false;
	}

	COutput(const COutput& output) {
		in = output.in;
		is_gold = output.is_gold;
	}
};

#endif /* CState_H_ */