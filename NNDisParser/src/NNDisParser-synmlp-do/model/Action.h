#ifndef CAction_H_
#define CAction_H_

#include "HyperParams.h"

class CAction {
public:
	enum CODE { REDUCE = 0, SHIFT = 1, POP_ROOT = 2, NO_ACTION = 3 };
	enum NUCLEAR {NN = 0, NS = 1, SN = 2};

	short _label;
	short _nuclear;
	unsigned long _code;

	string _label_str;

	CAction() {
		_code = NO_ACTION;
		_label = -1;
		_nuclear = -1;
		_label_str = nullkey;
	}

	CAction(const int &code, const short &nuclear, const short &label) {
		_code = code;
		_label = label;
		_nuclear = nuclear;
	}

	inline bool isNone() const { return _code == NO_ACTION; }

	inline bool isFinish() const { return _code == POP_ROOT; }

	inline bool isShift() const { return _code == SHIFT; }

	inline bool isReduce() const { return _code == REDUCE; }

	inline std::string str(const HyperParams &opts) const {
		if (isShift())
			return "SHIFT";
		else if (isReduce()) {
			if(_nuclear == NN)
				return "REDUCE_NN_" + opts.labelAlpha.from_id(_label);
			if(_nuclear == NS)
				return "REDUCE_NS_" + opts.labelAlpha.from_id(_label);
			if(_nuclear == SN)
				return "REDUCE_SN_" + opts.labelAlpha.from_id(_label);
		}
		else if (isFinish())
			return "POP_ROOT";
		else
			return "NO_ACTION";
	}

	inline void clear() {
		_code = NO_ACTION;
		_nuclear = -1;
		_label = -1;
		_label_str = nullkey;
	}

	inline void set(const int &code, const short &nuclear, const short &label) {
		_code = code;
		_nuclear = nuclear;
		_label = label;
	}

	inline void set(const CAction &ac) {
		_code = ac._code;
		_nuclear = ac._nuclear;
		_label = ac._label;
		_label_str = ac._label_str;
	}

	bool operator == (const CAction &action) const {
		return _code == action._code && _nuclear == action._nuclear && _label == action._label;
	}

	bool operator != (const CAction &action) const {
		return _code != action._code || _nuclear != action._nuclear || _label != action._label;
	}
};


#endif /* CAction_H_ */