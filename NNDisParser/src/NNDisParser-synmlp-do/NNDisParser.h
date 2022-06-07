#include "N3LDG.h"
#include "Options.h"
#include "Pipe.h"
#include "Utf.h"
#include "Driver.h"
#include "State.h"

class DisParser {
public:
	DisParser();
	virtual ~DisParser();

public:
	Driver m_driver;
	Options m_options;
	Pipe m_pipe;
	
	vector<BucketNode> extern_nodes;
	int node_count;
public:
	void createAlphabet(const vector<Instance> &vecInsts);
	void addTestAlpha(const vector<Instance> &vecInsts);
public:
	void train(const string &trainFile, const string &devFile, const string &testFile,
		const string &modelFile, const string &optionFile);
	void predict(const vector<Instance> &input, vector<CResult> &output);
	void test(const string &testFile, const string &outputFile, const string &modelFile);
	void getGoldActions(vector<Instance> &vecInsts);
	void train(const string &trainFile, const string &devFile,
		const string &testFile, const string &modelFile, const string &optionFile, const string &conllFolder);

	void getDepFeats(vector<Instance> &vecInsts, const string &path);
	void getSynFeats(vector<Instance> &vecInsts, const string &folder);
public:
	void writeModelFile(const string &outputModelFile);
	void loadModelFile(const string &inputModelFile);
	void test(const string &testFile, const string &outputFile, const string &modelFile, const string&optionFile);
};
