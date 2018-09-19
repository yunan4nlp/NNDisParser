#ifndef ATOM_FEATURE_H 
#define ATOM_FEATURE_H

class CNode {
public:
	short nuclear;
	short label;

	short edu_start;
	short edu_end;

	bool is_validate;

	CNode(){
		clear();
	}

	void clear(){
		nuclear = -1;
		label = -1;
		edu_start = -1;
		edu_end = -1;
		is_validate = false;
	}

	CNode(const CNode &other){
		nuclear = other.nuclear;
		label = other.label; 

		edu_start = other.edu_start;
		edu_end = other.edu_end;

		is_validate = other.is_validate;
	}

	string nuclear_str() {
		if (nuclear == CAction::NN)
			return "NN";
		else if (nuclear == CAction::SN)
			return "SN";
		else if (nuclear == CAction::NS)
			return "NS";
		else
			return nullkey;
	}

	string relation_str(const HyperParams &opts){
		if (label == -1)
			return nullkey;
		else
			return opts.labelAlpha.from_id(label);
	}
};

class AtomFeat {
public:
	vector<ConcatNode> *_pedu_lstm;

	int _next_index;
	string _pre_action_str;
	string _pre_pre_action_str;

	IncLSTM1Builder *_pre_action_lstm;

	CNode *_stack_top_0;
	CNode *_stack_top_1;
	CNode *_stack_top_2;

	string _nuclear_top_0;
	string _nuclear_top_1;
	string _nuclear_top_2;
	
	int _stack_size;
	int _buffer_size;
};
#endif /*ATOM_FEATURE_H */