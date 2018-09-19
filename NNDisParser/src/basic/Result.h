#ifndef BASIC_RESULT_H
#define BASIC_RESULT_H

#include <string>
#include <vector>
#include <fstream>
#include <string>
#include "N3LDG.h"
#include "Alphabet.h"
#include "Utf.h"

const static string NUCLEAR = "NUCLEAR";
const static string SATELLITE = "SATELLITE";
const static string SPAN = "span";

class SubTree{

public:
	string nuclear;
	string relation;
	int edu_start;
	int edu_end;
	
public:
	SubTree() {
		clear();
	}

	void clear() {
		nuclear = nullkey;
		relation = nullkey;
		edu_start = -1;
		edu_end = -1;
	}

	SubTree(const SubTree& tree) {
		nuclear = tree.nuclear;
		relation = tree.relation;
		edu_start = tree.edu_start;
		edu_end = tree.edu_end;
	}

	bool spanEqual(const SubTree& tree) const {
		return 
			edu_start == tree.edu_start && 
			edu_end == tree.edu_end;
	}

	bool nuclearEqual(const SubTree& tree) const {
		return 
			edu_start == tree.edu_start && 
			edu_end == tree.edu_end && 
			nuclear == tree.nuclear;
	}

	bool relationEqual(const SubTree& tree) const {
		return
			edu_start == tree.edu_start &&
			edu_end == tree.edu_end &&
			relation == tree.relation;
	}

	bool fullEqual(const SubTree& tree) const {
		return
			edu_start == tree.edu_start &&
			edu_end == tree.edu_end &&
			nuclear == tree.nuclear &&
			relation == tree.relation;
	}
};

class CResult {
public:
	vector<SubTree> subtrees;

public:
  inline void clear() {
    //words = nullptr;
		subtrees.clear();
 }

  inline void allocate(const int &size) {
		subtrees.resize(size);
  }

  inline int size() const {
    return subtrees.size();
  }

  inline void copyValuesFrom(const CResult &result) {
    static int size;
    size = result.size();
		int tree_size = result.subtrees.size();
    allocate(size);
		for(int i = 0;  i < tree_size; i++) {
			subtrees[i] = result.subtrees[i];
		}
  }

  inline void copyValuesFrom(const CResult &result, const vector<string> *pwords) {
    static int size;
    size = result.size();

		int tree_size = result.subtrees.size();
    allocate(size);
		for(int i = 0;  i < tree_size; i++) {
			subtrees[i] = result.subtrees[i];
		}
//    words = pwords;
  }

  inline std::string str() const {
    for (int i = 0; i < size(); ++i) {
      //std::cout << (*words)[i] << " " << tags[i] << " " << heads[i] << " " << labels[i] << std::endl;
      //std::cout << words[i] << " " << tags[i] << " " << heads[i] << " " << labels[i] << std::endl;
    }
  }

};


#endif
