#ifndef BASIC_INSTANCE_H
#define BASIC_INSTANCE_H

#include <stack>

#include "N3LDG.h"
#include "Metric.h"
#include "Action.h"
#include "Result.h"



class EDU {
public:
	int start_index;
	int end_index;

	string etype;

	vector<string> words;
	vector<string> tags;


	EDU() {
		start_index = -1;
		end_index = -1;
		etype = nullkey;
		words.clear();
		tags.clear();
	}
};

class DepFeat{
public:
	vector<string> words;
	vector<string> tags;
	vector<int> heads;
	vector<string> dep_relations;

	void resize(int size) {
		words.resize(size);
		tags.resize(size);
		heads.resize(size);
		dep_relations.resize(size);
	}

	void clear(){
		words.clear();
		tags.clear();
		heads.clear();
		dep_relations.clear();
	}

	DepFeat() {
		clear();
	}

	DepFeat(const DepFeat &other) {
		words = other.words;
		tags = other.tags;
		heads = other.heads;
		dep_relations = other.dep_relations;
	}

};

class Instance {
public:
	Instance() {
	}

	Instance(const Instance &other) {
		copyValuesFrom(other);
	}

	~Instance() {
	}

public:

	int size() const {
		return words.size();
	}


	void clear() {
		words.clear();
		result.clear();
		tags.clear();
		edus.clear();
		sent_types.clear();

		total_text.clear();
		total_tags.clear();

		gold_actions.clear();
		clearVec(syn_feats);
	}

	void copyValuesFrom(const Instance &anInstance) {
		edus = anInstance.edus;
		sent_types = anInstance.sent_types;
		words = anInstance.words;
		tags = anInstance.tags;
		total_text = anInstance.total_text;
		total_tags = anInstance.total_tags;

		gold_actions = anInstance.gold_actions;

		result.copyValuesFrom(anInstance.result, &words);
		syn_feats = anInstance.syn_feats;
		dep_feats = anInstance.dep_feats;
	}

	void evaluate(CResult &other, Metric &span, Metric &nuclear, Metric &relation, Metric &full) const {
		span.overall_label_count += result.subtrees.size();
		span.predicated_label_count += other.subtrees.size();
		for (int idx = 0; idx < other.subtrees.size(); idx++) {
			for (int idy = 0; idy < result.subtrees.size(); idy++) {
				if (other.subtrees[idx].spanEqual(result.subtrees[idy])) {
					span.correct_label_count++;
					break;
				}
			}
		}

		nuclear.overall_label_count += result.subtrees.size();
		nuclear.predicated_label_count += other.subtrees.size();
		for (int idx = 0; idx < other.subtrees.size(); idx++) {
			for (int idy = 0; idy < result.subtrees.size(); idy++) {
				if (other.subtrees[idx].nuclearEqual(result.subtrees[idy])) {
					nuclear.correct_label_count++;
					break;
				}
			}
		}

		relation.overall_label_count += result.subtrees.size();
		relation.predicated_label_count += other.subtrees.size();
		for (int idx = 0; idx < other.subtrees.size(); idx++) {
			for (int idy = 0; idy < result.subtrees.size(); idy++) {
				if (other.subtrees[idx].relationEqual(result.subtrees[idy])) {
					relation.correct_label_count++;
					break;
				}
			}
		}

		full.overall_label_count += result.subtrees.size();
		full.predicated_label_count += other.subtrees.size();
		for (int idx = 0; idx < other.subtrees.size(); idx++) {
			for (int idy = 0; idy < result.subtrees.size(); idy++) {
				if (other.subtrees[idx].fullEqual(result.subtrees[idy])) {
					full.correct_label_count++;
					break;
				}
			}
		}
	}

	void parse_tree(const string &tree_str) {
		vector<string> buffer;
		vector<pair<int, int> > subtree_stack;
		stack<string> op_stack;
		stack<string> relation_stack;
		stack<string> action_stack;
		split_bychar(tree_str, buffer, ' ');
		int step = 0;
		string start, end;
		int edu_start, edu_end;
		int buffer_size = buffer.size(), node_size;
		while (true) {
			assert(step <= buffer_size);
			if (step == buffer_size)
				break;
			if (buffer[step] == "(") {
				op_stack.push(buffer[step]);
				relation_stack.push(buffer[step + 1]);
				action_stack.push(buffer[step + 2]);
				if (buffer[step + 2] == "t") {
					start = buffer[step + 3];
					end = buffer[step + 4];
					step += 2;
				}
				step += 3;
			}
			else if (buffer[step] == ")") {
				const string &action = action_stack.top();
				if (action == "t") {
					EDU edu;
					edu.start_index = atoi(start.c_str());
					edu.end_index = atoi(end.c_str());
					for (int i = 0; i < sent_types.size(); i++) {
						if (edu.start_index >= sent_types[i].first.first &&
							edu.end_index <= sent_types[i].first.second) {
							edu.etype = sent_types[i].second;
							break;
						}
					}
					edu_start = edus.size();
					edu_end = edus.size();
					subtree_stack.push_back(make_pair(edu_start, edu_end));
					edus.push_back(edu);
					CAction ac;
					ac._code = CAction::SHIFT;
					ac._label_str = relation_stack.top();
					assert(relation_stack.top() == "leaf");
					gold_actions.push_back(ac);
				}
				else if (action == "l" || action == "r" || action == "c") {
					CAction ac;
					ac._code = CAction::REDUCE;

					if (action == "l")
						ac._nuclear = CAction::NS;
					else if(action == "r")
						ac._nuclear = CAction::SN;
					else if(action == "c")
						ac._nuclear = CAction::NN;
					ac._label_str = relation_stack.top();
					gold_actions.push_back(ac);


					int subtree_size = subtree_stack.size();
					assert(subtree_size >= 2);

					pair<int, int> &right_tree_index = subtree_stack[subtree_size - 1];
					pair<int, int> &left_tree_index = subtree_stack[subtree_size - 2];
					SubTree right_tree;
					SubTree left_tree;
					right_tree.edu_start = right_tree_index.first;
					right_tree.edu_end = right_tree_index.second;

					left_tree.edu_start = left_tree_index.first;
					left_tree.edu_end = left_tree_index.second;
					if (action == "l") {
						left_tree.nuclear = NUCLEAR;
						right_tree.nuclear = SATELLITE;
						left_tree.relation = SPAN;
						right_tree.relation = ac._label_str;
					}
					else if(action == "r") {
						left_tree.nuclear = SATELLITE;
						right_tree.nuclear = NUCLEAR;
						left_tree.relation = ac._label_str;
						right_tree.relation = SPAN;
					}
					else if(action == "c") {
						left_tree.nuclear = NUCLEAR;
						right_tree.nuclear = NUCLEAR;
						left_tree.relation = ac._label_str;
						right_tree.relation = ac._label_str;
					}

					result.subtrees.push_back(left_tree);
					result.subtrees.push_back(right_tree);

					edu_start = right_tree_index.first;
					edu_end = right_tree_index.second;
					assert(left_tree_index.second + 1 == edu_start);
					left_tree_index.second = edu_end;
					subtree_stack.pop_back();
				}
				action_stack.pop();
				relation_stack.pop();
				op_stack.pop();
				step += 1;
			}
		}
		CAction ac;
		ac._code = CAction::POP_ROOT;
		gold_actions.push_back(ac);
		// check stack.
		assert(op_stack.empty() && relation_stack.empty() && action_stack.empty());
		int edu_size = edus.size();
		int total_text_size = total_text.size();
		int total_tag_size = total_tags.size();
		int type_size = sent_types.size();
		int word_size = words.size();
		// check word and tag num
		assert(total_tag_size == total_text_size);
		assert(word_size + type_size == total_tag_size);
		int sum = 0;
		// check edu.
		for (int idx = 0; idx < edu_size; idx++) {
			EDU &edu = edus[idx];
			assert(edu.start_index <= edu.end_index);
			assert(edu.start_index >= 0 && edu.end_index < total_text_size);
			if (idx != edu_size - 1) {
				assert(edu.end_index + 1 == edus[idx + 1].start_index);
			}
			for (int idy = edu.start_index; idy <= edu.end_index; idy++) {
				if(total_tags[idy] != nullkey){
					edu.words.push_back(normalize_to_lowerwithdigit(total_text[idy]));
					edu.tags.push_back(total_tags[idy]);
				}
			}
			assert(edu.words.size() == edu.tags.size());
			sum += edu.words.size();
		}
		// check subtree
		assert(sum == words.size() && result.subtrees.size() + 2 == edus.size() * 2);
		int subtree_size = result.subtrees.size();
		for(int idx = 0; idx < subtree_size; idx++) {
			const SubTree &subtree = result.subtrees[idx];
			assert(subtree.relation != nullkey && subtree.nuclear != nullkey);
		}
	}

public:
	vector<vector<BucketNode*> > syn_feats;
	vector<EDU> edus;
	vector<pair<pair<int, int>, string > > sent_types;
	vector<string> words;
	vector<string> tags;
	vector<string> total_text;
	vector<string> total_tags;

	vector<CAction> gold_actions;

	vector<DepFeat> dep_feats;
	
	CResult result;
};

#endif
