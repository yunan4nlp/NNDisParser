#include "Argument_helper.h"
#include "NNDisParser.h"
#include <chrono>


DisParser::DisParser() {
	srand(0);
}

DisParser::~DisParser() {
}

void DisParser::createAlphabet(const vector<Instance>& vecInsts) {
	cout << "Creating Alphabet..." << endl;
	int maxsize = vecInsts.size();
	int word_size;
	int gold_size;
	for (int idx = 0; idx < maxsize; idx++) {
		const Instance &inst = vecInsts[idx];
		word_size = inst.words.size();
		for (int idy = 0; idy < word_size; idy++) {
			m_driver._hyperparams.word_stat[inst.words[idy]]++;
			m_driver._hyperparams.tag_stat[inst.tags[idy]]++;
		}
		gold_size = inst.gold_actions.size();
		for (int idy = 0; idy < gold_size; idy++) {
			const CAction &ac = inst.gold_actions[idy];
			if (!ac.isShift() && !ac.isFinish()) {
				m_driver._hyperparams.labelAlpha.from_string(ac._label_str);
			}
		}
		
	}
	m_driver._hyperparams.labelAlpha.from_string(nullkey);

	m_driver._hyperparams.word_stat[unknownkey] = m_options.wordCutOff + 1;
	m_driver._hyperparams.wordAlpha.initial(m_driver._hyperparams.word_stat, m_options.wordCutOff);
	m_driver._hyperparams.tagAlpha.initial(m_driver._hyperparams.tag_stat, 0);

	m_driver._hyperparams.wordAlpha.set_fixed_flag(true);
	m_driver._hyperparams.tagAlpha.set_fixed_flag(true);
	m_driver._hyperparams.labelAlpha.set_fixed_flag(true);

	cout << "word alpha size : " << m_driver._hyperparams.wordAlpha.size() << endl;
	cout << "tag alpha size : " << m_driver._hyperparams.tagAlpha.size() << endl;
	cout << "label alpha size : " << m_driver._hyperparams.labelAlpha.size() << endl;
}

void DisParser::addTestAlpha(const vector<Instance> &vecInsts) {
	cout << "Add Test Alphabet..." << endl;
	int maxsize = vecInsts.size();
	int word_size;
	for (int idx = 0; idx < maxsize; idx++) {
		const Instance &inst = vecInsts[idx];
		word_size = inst.words.size();
		for (int idy = 0; idy < word_size; idy++) {
			if(!m_options.wordFineTune)m_driver._hyperparams.word_stat[inst.words[idy]]++;
		}
	}
	cout << "word stat: " << m_driver._hyperparams.word_stat.size() << endl;
}


void DisParser::getGoldActions(vector<Instance> &vecInsts) {
	int inst_size = vecInsts.size();
	int gold_action_size;
	CResult result;
	Metric span, nuclear, relation, full;
	span.reset();
	nuclear.reset();
	relation.reset();
	full.reset();
	int shift_num = 0;
	int reduce_nn_num = 0, reduce_ns_num = 0, reduce_sn_num = 0;
	int edu_size;
	for(int idx = 0; idx < inst_size; idx++) { 
		Instance &inst = vecInsts[idx];
		gold_action_size = inst.gold_actions.size();
		vector<CAction> &gold_action = inst.gold_actions;
		for(int idy = 0; idy < gold_action_size; idy++) { 
			edu_size = inst.edus.size();
			for(int idy = 0; idy < edu_size; idy++) {
				m_driver._hyperparams.etype_stat[inst.edus[idy].etype]++;
			}
			CAction &ac = gold_action[idy];
			if (ac.isShift())
				shift_num++;
			if(ac.isReduce()) {
				if (ac._nuclear == CAction::NN)
					reduce_nn_num++;
				else if (ac._nuclear == CAction::NS)
					reduce_ns_num++;
				else if (ac._nuclear == CAction::SN)
					reduce_sn_num++;
				else {
					cerr << "reduce error" << endl;
				}
			}
			if (ac._code == CAction::REDUCE) {
				ac._label = m_driver._hyperparams.labelAlpha.from_string(ac._label_str);
				assert(ac._label != -1);
			}
			m_driver._hyperparams.action_stat[ac.str(m_driver._hyperparams)]++;
		}
	}
	cout <<"Reduce NN: " <<  reduce_nn_num << endl;
	cout <<"Reduce NS: " <<  reduce_ns_num << endl;
	cout <<"Reduce SN: " <<  reduce_sn_num << endl;
	cout << "Shift : " << shift_num << endl;
	m_driver._hyperparams.action_stat[nullkey] = 1;
	m_driver._hyperparams.actionAlpha.initial(m_driver._hyperparams.action_stat, 0);
	m_driver._hyperparams.actionAlpha.set_fixed_flag(true);
	m_driver._hyperparams.actionNum = m_driver._hyperparams.actionAlpha.size();

	m_driver._hyperparams.etypeAlpha.initial(m_driver._hyperparams.etype_stat, 0);
	m_driver._hyperparams.etypeAlpha.set_fixed_flag(true);

	vector<CState> all_states(m_driver._hyperparams.maxStateSize);
	int step;
	for (int idx = 0; idx < inst_size; idx++) {
		const Instance &inst = vecInsts[idx];
		const vector<CAction> &gold_actions = inst.gold_actions;
		int action_size = gold_actions.size();
		all_states[0].clear();
		all_states[0].ready(&inst);
		step = 0;
		while (!all_states[step].isEnd()) {
			assert(step < action_size);
			all_states[step].move(&all_states[step + 1], gold_actions[step]);
			step++;
		}
		assert(step == action_size);
		all_states[step].getResults(result, m_driver._hyperparams);
		inst.evaluate(result, span, nuclear, relation, full);
		if (!span.bIdentical() || !nuclear.bIdentical() || !relation.bIdentical() || !full.bIdentical()) {
			std::cout << "error state conversion!" << std::endl;
			exit(0);
		}
	}
}

void DisParser::train(const string &trainFile, const string &devFile, 
	const string &testFile, const string &modelFile, const string &optionFile) {
	if (optionFile != "")
		m_options.load(optionFile);
	m_options.showOptions();
	m_driver._hyperparams.setRequared(m_options);

	vector<Instance> trainInsts, devInsts, testInsts;
	m_pipe.readInstances(trainFile, trainInsts, m_options.maxInstance);
	m_pipe.readInstances(devFile, devInsts, m_options.maxInstance);
	m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);
	
	getDepFeats(trainInsts, m_options.conllFolder + path_separator + "train.conll.predict");
	getDepFeats(devInsts, m_options.conllFolder + path_separator + "dev.conll.predict");
	getDepFeats(testInsts, m_options.conllFolder + path_separator + "test.conll.predict");

	int word_count = 0, max_size;
	max_size = trainInsts.size();
	for (int idx = 0; idx < max_size; idx++) {
		word_count += trainInsts[idx].words.size();
	}
	max_size = devInsts.size();
	for (int idx = 0; idx < max_size; idx++) {
		word_count += devInsts[idx].words.size();
	}
	max_size = testInsts.size();
	for (int idx = 0; idx < max_size; idx++) {
		word_count += testInsts[idx].words.size();
	}
	extern_nodes.resize(word_count * 10);
	node_count = 0;

	string syn = "conll.dump.results";
	getSynFeats(devInsts, m_options.dumpFolder + path_separator + "dev." + syn);
	getSynFeats(testInsts, m_options.dumpFolder + path_separator + "test." + syn);
	
	vector<vector<Instance> > otherInsts(m_options.testFiles.size());
	for (int idx = 0; idx < m_options.testFiles.size(); idx++) {
		m_pipe.readInstances(m_options.testFiles[idx], otherInsts[idx], m_options.maxInstance);
	}

	addTestAlpha(devInsts);
	addTestAlpha(testInsts);
	createAlphabet(trainInsts);
	getGoldActions(trainInsts);

	if(m_options.wordEmbFile == "")
		m_driver._modelparams.edu_params.word_table.initial(&m_driver._hyperparams.wordAlpha, m_options.wordEmbSize, m_options.wordFineTune);
	else
		m_driver._modelparams.edu_params.word_table.initial(&m_driver._hyperparams.wordAlpha, m_options.wordEmbFile, m_options.wordFineTune);
	m_driver._hyperparams.wordDim = m_driver._modelparams.edu_params.word_table.nDim;

	m_driver._modelparams.edu_params.tag_table.initial(&m_driver._hyperparams.tagAlpha, m_options.tagEmbSize, m_options.tagFineTune);
	m_driver._hyperparams.tagDim = m_driver._modelparams.edu_params.tag_table.nDim;

	m_driver._hyperparams.wordConcatDim = m_driver._hyperparams.wordDim + m_driver._hyperparams.tagDim;

	m_driver._modelparams.etype_table.initial(&m_driver._hyperparams.etypeAlpha, m_options.etypeEmbSize, m_options.etypeFineTune);
	m_driver._hyperparams.etypeDim = m_driver._modelparams.etype_table.nDim;

	m_driver._hyperparams.eduConcatDim = m_driver._hyperparams.eduHiddenDim + m_driver._hyperparams.etypeDim;
	//m_driver._hyperparams.eduConcatDim = m_driver._hyperparams.eduHiddenDim;

	m_driver.initial();
	double bestFmeasure = -1;
	int inputSize = trainInsts.size();
	std::vector<int> indexes;
	for (int i = 0; i < inputSize; ++i)
		indexes.push_back(i);
	int devNum = devInsts.size(), testNum = testInsts.size();
	int batchBlock = inputSize / m_options.batchSize;
	if (inputSize % m_options.batchSize != 0)
		batchBlock++;
	
	vector<Instance> subInstances;
	Metric eval;
	for (int iter = 0; iter < m_options.maxIter; ++iter) {
		std::cout << "##### Iteration " << iter + 1 << std::endl;
		random_shuffle(indexes.begin(), indexes.end());
		eval.reset();
		std::cout << "random: " << indexes[0] << ", " << indexes[indexes.size() - 1] << std::endl;


		if (iter + 1 >= m_options.startDecay) {
			dtype adaAlpha = m_options.adaAlpha / (1 + m_options.decay * iter);
			m_driver.setUpdateParameters(m_options.regParameter, adaAlpha, m_options.adaEps);
		} 

		if (iter + 1 >= m_options.startBeam) {
			m_driver.setGraph(true);
		}

		if(iter + 1 >= m_options.startDynamicOracle){
			m_driver._hyperparams.dynamicOracle = true;
			cout << "Dynamic Orcale" << endl;
		}

		if (m_driver._useBeam) {
			cout << "Beam Search" << endl;
			cout << "Beam Alpha: " << m_driver._beam_ada._alpha << endl;
			m_driver.setDropFactor(1);
		} else {
			cout << "Greedy Search" << endl;
			cout << "Greedy Alpha: " << m_driver._ada._alpha << endl;
		}

		if (m_options.reachDrop > 0 && !m_driver._useBeam)
			m_driver.setDropFactor(iter * 1.0 / m_options.reachDrop);

		auto t_start = std::chrono::high_resolution_clock::now();
		for (int idx = 0; idx < batchBlock; idx++) {
			int start_pos = idx * m_options.batchSize;
			int end_pos = (idx + 1) * m_options.batchSize;
			if (end_pos > inputSize)
				end_pos = inputSize;
			subInstances.clear();
			for (int idy = start_pos; idy < end_pos; idy++) { // one batch
				subInstances.push_back(trainInsts[indexes[idy]]);
			}
			dtype cost = m_driver.train(subInstances);
			eval.overall_label_count += m_driver._eval.overall_label_count;
			eval.correct_label_count += m_driver._eval.correct_label_count;
			if ((idx + 1) % (m_options.verboseIter) == 0) {
				auto t_end = std::chrono::high_resolution_clock::now();
				std::cout << "current: " << idx + 1 << ", Cost = " << cost << ", Correct(%) = " << eval.getAccuracy()
					<< ", Time: " << std::chrono::duration<double>(t_end - t_start).count() << "s" << std::endl;
			}
			m_driver.updateModel();
		}

		bool bCurIterBetter;
		vector<CResult> decodeInstResults;
		Metric dev_span, dev_nuclear, dev_relation, dev_full;
		Metric test_span, test_nuclear, test_relation, test_full;

		if (devNum > 0) {
			auto t_start_dev = std::chrono::high_resolution_clock::now();
			cout << "Dev start." << std::endl;
			bCurIterBetter = false;
			if (!m_options.outBest.empty()) {
				decodeInstResults.clear();
			}
			dev_span.reset();
			dev_nuclear.reset();
			dev_relation.reset();
			predict(devInsts, decodeInstResults);
			int devNum = devInsts.size();
			for (int idx = 0; idx < devNum; idx++) {
				devInsts[idx].evaluate(decodeInstResults[idx], dev_span, dev_nuclear, dev_relation, dev_full);
			}
			auto t_end_dev = std::chrono::high_resolution_clock::now();
			cout << "Dev finished. Total time taken is: " << std::chrono::duration<double>(t_end_dev - t_start_dev).count() << std::endl;
			cout << "dev:" << std::endl;
			cout << "S: ";
			dev_span.print();
			cout << "N: ";
			dev_nuclear.print();
			cout << "R: ";
			dev_relation.print();
			cout << "F: ";
			dev_full.print();
			if (!m_options.outBest.empty() && dev_full.getAccuracy() > bestFmeasure) {
				m_pipe.outputAllInstances(devFile + m_options.outBest + to_string(iter), decodeInstResults);
				bCurIterBetter = true;
			}

			if (testNum > 0) {
				auto t_start_test = std::chrono::high_resolution_clock::now();
				cout << "Test start." << std::endl;
				if (!m_options.outBest.empty()) {
					decodeInstResults.clear();
				}
				test_span.reset();
				test_nuclear.reset();
				test_relation.reset();
				test_full.reset();
				predict(testInsts, decodeInstResults);
				for (int idx = 0; idx < testInsts.size(); idx++) {
					testInsts[idx].evaluate(decodeInstResults[idx], test_span, test_nuclear, test_relation, test_full);
				}
				auto t_end_test = std::chrono::high_resolution_clock::now();
				cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_test - t_start_test).count() << std::endl;
				cout << "test:" << std::endl;
				cout << "S: ";
				test_span.print();
				cout << "N: ";
				test_nuclear.print();
				cout << "R: ";
				test_relation.print();
				cout << "F: ";
				test_full.print();

				if (!m_options.outBest.empty() && bCurIterBetter) {
					m_pipe.outputAllInstances(testFile + m_options.outBest + to_string(iter), decodeInstResults);
				}
			}
			if (m_options.saveIntermediate && dev_full.getAccuracy() > bestFmeasure) {
				std::cout << "Exceeds best previous performance of " << bestFmeasure << ". Saving model file.." << std::endl;
				bestFmeasure = dev_full.getAccuracy();
				writeModelFile(modelFile);
			}
		}
	}
}

void DisParser::test(const string &testFile, const string &outputFile, const string &modelFile, const string&optionFile) {
	if (optionFile != "")
	m_options.load(optionFile);
	m_options.showOptions();
	m_driver._hyperparams.setRequared(m_options);

	loadModelFile(modelFile);

	vector<Instance> testInsts;
	m_pipe.readInstances(testFile, testInsts, m_options.maxInstance);
	getDepFeats(testInsts, testFile + ".conll");

	int word_count = 0, max_size;
	max_size = testInsts.size();
	for (int idx = 0; idx < max_size; idx++) {
		word_count += testInsts[idx].words.size();
	}
	extern_nodes.resize(word_count * 10);
	node_count = 0;

	getSynFeats(testInsts, testFile + ".dump");

	/* 
		NOTE (Mat-sipahi): The following lines (down to `m_driver.initial`) are copied from `train` function 
	          in order to avoid segmentations faults we used to get because if uninitalized hyperparams in different parts of the code.
			  But I'm not sure which one of them is redundant or coming in incorrect order.
	*/
	addTestAlpha(testInsts);
	//createAlphabet(trainInsts);
	//getGoldActions(trainInsts);

    //if(m_options.wordEmbFile == "") {
	//	m_driver._modelparams.edu_params.word_table.initial(&m_driver._hyperparams.wordAlpha, m_options.wordEmbSize, m_options.wordFineTune);
	//}
	//else
	//	m_driver._modelparams.edu_params.word_table.initial(&m_driver._hyperparams.wordAlpha, m_options.wordEmbFile, m_options.wordFineTune);
	m_driver._hyperparams.wordDim = m_driver._modelparams.edu_params.word_table.nDim;
	m_driver._modelparams.edu_params.tag_table.initial(&m_driver._hyperparams.tagAlpha, m_options.tagEmbSize, m_options.tagFineTune);
	m_driver._hyperparams.tagDim = m_driver._modelparams.edu_params.tag_table.nDim;

	m_driver._hyperparams.wordConcatDim = m_driver._hyperparams.wordDim + m_driver._hyperparams.tagDim;
	//m_driver._modelparams.etype_table.initial(&m_driver._hyperparams.etypeAlpha, m_options.etypeEmbSize, m_options.etypeFineTune);
	m_driver._hyperparams.etypeDim = m_driver._modelparams.etype_table.nDim;
	m_driver._hyperparams.eduConcatDim = m_driver._hyperparams.eduHiddenDim + m_driver._hyperparams.etypeDim;
	//m_driver._hyperparams.eduConcatDim = m_driver._hyperparams.eduHiddenDim;

	m_driver._modelparams.edu_params.word_table.elems = &m_driver._hyperparams.wordAlpha;
	m_driver._modelparams.edu_params.tag_table.elems = &m_driver._hyperparams.tagAlpha;
	m_driver._modelparams.etype_table.elems = &m_driver._hyperparams.etypeAlpha;
	m_driver._modelparams.scored_action_table.elems = &m_driver._hyperparams.actionAlpha;
	
	//m_driver._hyperparams.actionAlpha.initial(m_driver._hyperparams.action_stat, 0);
	m_driver._hyperparams.actionNum = m_driver._hyperparams.actionAlpha.size();
	//m_driver._hyperparams.etypeAlpha.initial(m_driver._hyperparams.etype_stat, 0);

	m_driver.initial();

	/*
		End of the block copied from `train` (Mat-sipahi)
	*/

	vector<CResult> decodeInstResults;
	int testNum = testInsts.size();


	Metric test_span, test_nuclear, test_relation, test_full;
	if (testNum > 0) {
		auto t_start_test = std::chrono::high_resolution_clock::now();
		cout << "Test start." << std::endl;
		if (!m_options.outBest.empty()) {
			decodeInstResults.clear();
		}
		test_span.reset();
		test_nuclear.reset();
		test_relation.reset();
		test_full.reset();

		predict(testInsts, decodeInstResults);

		for (int idx = 0; idx < testInsts.size(); idx++) {
			testInsts[idx].evaluate(decodeInstResults[idx], test_span, test_nuclear, test_relation, test_full);
		}
		auto t_end_test = std::chrono::high_resolution_clock::now();
		cout << "Test finished. Total time taken is: " << std::chrono::duration<double>(t_end_test - t_start_test).count() << std::endl;
		cout << "test:" << std::endl;
		cout << "S: ";
		test_span.print();
		cout << "N: ";
		test_nuclear.print();
		cout << "R: ";
		test_relation.print();
		cout << "F: ";
		test_full.print();

		m_pipe.outputAllInstances(outputFile, decodeInstResults);
	}
}

void DisParser::getDepFeats(vector<Instance> &vecInsts, const string &path) {
	ifstream file(path.c_str());
	string line;
	vector<string> vecLines;
	vector<string> info;
	int index = 0, sent_size;
	while (getline(file, line)) {
		if (line == "") {
			sent_size = vecLines.size();
			DepFeat dep_feat;
			dep_feat.resize(sent_size);
			for (int idx = 0; idx < sent_size; idx++) {
				split_bychar(vecLines[idx], info, '\t');
				dep_feat.words[idx] = normalize_to_lowerwithdigit(info[1]);
				dep_feat.tags[idx] = info[3];
				dep_feat.heads[idx] = atoi(info[6].c_str()) - 1;
				dep_feat.dep_relations[idx] = info[7];
			}
			Instance &cur_inst = vecInsts[index];
			cur_inst.dep_feats.push_back(dep_feat);
			if (cur_inst.sent_types.size() == cur_inst.dep_feats.size()) {
				index++;
				if (index == vecInsts.size())
					break;
			}
			vecLines.clear();
		}
		else
			vecLines.push_back(line);
	}
	
	file.close();
	// checking...
	int inst_size = vecInsts.size();
	int word_num, i, offset;

	for (int idx = 0; idx < inst_size; idx++) {
		const Instance &inst = vecInsts[idx];
		word_num = 0;
		int dep_feat_size = inst.dep_feats.size();
		for (int idy = 0; idy < dep_feat_size; idy++) {
			word_num += inst.dep_feats[idy].words.size();
		}
		assert(word_num == inst.words.size());
		i = 0, offset = 0;
		for (int idy = 0; idy < word_num; idy++) {
			const DepFeat &cur_feat = inst.dep_feats[i];
			assert(inst.words[idy] == cur_feat.words[idy - offset] &&
				inst.tags[idy] == cur_feat.tags[idy - offset]);
			if (idy - offset + 1 == cur_feat.words.size()) {
				i++;
				offset += cur_feat.words.size();
			}
		}
	}
}
  

void DisParser::getSynFeats(vector<Instance> &vecInsts, const string &folder) {
	string file1 = folder + path_separator + "arc_dep";
	string file2 = folder + path_separator + "rel_head";
	string file3 = folder + path_separator + "arc_head";
	string file4 = folder + path_separator + "rel_dep";
	string file5 = folder + path_separator + "lstm_out";

	ifstream inf1(file1.c_str());
	ifstream inf2(file2.c_str());
	ifstream inf3(file3.c_str());
	ifstream inf4(file4.c_str());
	ifstream inf5(file5.c_str());
	string strLine;
	vector<string> vecLine1, vecLine2, vecLine3, vecLine4, vecLine5;
	vector<string> vecInfo;
	int index = 0, num = 0, offset = 0, word_offset;
	while (1) {
		vecLine1.clear();
		while (1) {
			if (!my_getline(inf1, strLine)) {
				break;
			}
			if (strLine.empty())
				break;
			vecLine1.push_back(strLine);
		}

		vecLine2.clear();
		while (1) {
			if (!my_getline(inf2, strLine)) {
				break;
			}
			if (strLine.empty())
				break;
			vecLine2.push_back(strLine);
		}

		vecLine3.clear();
		while (1) {
			if (!my_getline(inf3, strLine)) {
				break;
			}
			if (strLine.empty())
				break;
			vecLine3.push_back(strLine);
		}

		vecLine4.clear();
		while (1) {
			if (!my_getline(inf4, strLine)) {
				break;
			}
			if (strLine.empty())
				break;
			vecLine4.push_back(strLine);
		}

		vecLine5.clear();
		while (1) {
			if (!my_getline(inf5, strLine)) {
				break;
			}
			if (strLine.empty())
				break;
			vecLine5.push_back(strLine);
		}

		int vec_size = vecLine1.size();
		if (vec_size == 0) {
			if (index != vecInsts.size()) {
				std::cout << "some instances do not have external features: " << index << ":" << vecInsts.size() << std::endl;
			}
			break;
		}
		if (vecLine2.size() != vec_size || vecLine3.size() != vec_size ||
			vecLine4.size() != vec_size || vecLine5.size() != vec_size) {
			std::cout << "extern feature input error" << std::endl;
		}
		Instance &cur_inst = vecInsts[index];
		const DepFeat &cur_dep_feat = cur_inst.dep_feats[num - offset];
		if (num - offset == 0)
			word_offset = 0;

		if (cur_dep_feat.words.size() != vec_size) {
			continue;
		}

		for (int i = 0; i < vec_size; i++) {
			vecInfo.clear();
			split_bychar(vecLine1[i], vecInfo, ' ');

			assert(normalize_to_lowerwithdigit(vecInfo[0]).compare(cur_dep_feat.words[i]) == 0);
		}
		int cur_word_size = cur_dep_feat.words.size(), syn_offset;
		for (int i = 0; i < cur_word_size; i++) {
			syn_offset = i + word_offset;
			cur_inst.syn_feats[syn_offset].resize(6);

			split_bychar(vecLine1[i], vecInfo, ' ');
			assert(vecInfo.size() == 501);
			extern_nodes[node_count].init(500, -1);
			cur_inst.syn_feats[syn_offset][0] = &extern_nodes[node_count];
			for (int j = 0; j < 500; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			assert(vecInfo.size() == 513);
			extern_nodes[node_count].init(512, -1);
			cur_inst.syn_feats[syn_offset][0] = &extern_nodes[node_count];
			for (int j = 0; j < 512; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			node_count++;

			split_bychar(vecLine2[i], vecInfo, ' ');
			assert(vecInfo.size() == 101);
			extern_nodes[node_count].init(100, -1);
			cur_inst.syn_feats[syn_offset][1] = &extern_nodes[node_count];
			for (int j = 0; j < 100; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			assert(vecInfo.size() == 129);
			extern_nodes[node_count].init(128, -1);
			cur_inst.syn_feats[syn_offset][1] = &extern_nodes[node_count];
			for (int j = 0; j < 128; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			node_count++;

			split_bychar(vecLine3[i], vecInfo, ' ');
			assert(vecInfo.size() == 501);
			extern_nodes[node_count].init(500, -1);
			cur_inst.syn_feats[syn_offset][2] = &extern_nodes[node_count];
			for (int j = 0; j < 500; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			assert(vecInfo.size() == 513);
			extern_nodes[node_count].init(512, -1);
			cur_inst.syn_feats[syn_offset][2] = &extern_nodes[node_count];
			for (int j = 0; j < 512; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			node_count++;

			split_bychar(vecLine4[i], vecInfo, ' ');
			assert(vecInfo.size() == 101);
			extern_nodes[node_count].init(100, -1);
			cur_inst.syn_feats[syn_offset][3] = &extern_nodes[node_count];
			for (int j = 0; j < 100; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			assert(vecInfo.size() == 129);
			extern_nodes[node_count].init(128, -1);
			cur_inst.syn_feats[syn_offset][3] = &extern_nodes[node_count];
			for (int j = 0; j < 128; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			node_count++;

			split_bychar(vecLine5[i], vecInfo, ' ');
			assert(vecInfo.size() == 801);
			extern_nodes[node_count].init(400, -1);
			cur_inst.syn_feats[syn_offset][4] = &extern_nodes[node_count];
			for (int j = 0; j < 400; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			node_count++;
			extern_nodes[node_count].init(400, -1);
			cur_inst.syn_feats[syn_offset][5] = &extern_nodes[node_count];
			for (int j = 0; j < 400; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 401].c_str());
			}
			assert(vecInfo.size() == 1025);
			extern_nodes[node_count].init(512, -1);
			cur_inst.syn_feats[syn_offset][4] = &extern_nodes[node_count];
			for (int j = 0; j < 512; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 1].c_str());
			}
			node_count++;
			extern_nodes[node_count].init(512, -1);
			cur_inst.syn_feats[syn_offset][5] = &extern_nodes[node_count];
			for (int j = 0; j < 512; j++) {
				extern_nodes[node_count].val[j] = atof(vecInfo[j + 513].c_str());
			}
			node_count++;
		}

		int cur_feat_size = cur_inst.dep_feats.size();
		if(num - offset + 1 == cur_feat_size) {
			index++;
			offset += cur_feat_size;
		}

		word_offset += cur_dep_feat.words.size();
		num++;
	}

	inf1.close();
	inf2.close();
	inf3.close();
	inf4.close();
	inf5.close();
}


void DisParser::writeModelFile(const string &outputModelFile) {
	ofstream outf(outputModelFile.c_str());
	m_driver._hyperparams.write(outf);
	m_driver._modelparams.saveModel(outf);	
	outf.close();
}

void DisParser::loadModelFile(const string &inputModelFile) {
	ifstream inf(inputModelFile.c_str());
	m_driver._hyperparams.read(inf);
	m_driver._modelparams.loadModel(inf, m_driver._hyperparams);
	inf.close();
}

void DisParser::predict(const vector<Instance> &inputs, vector<CResult> &outputs) {
	vector<Instance> batch_input;
	vector<CResult> batch_outputs;
	int input_size = inputs.size();
	outputs.clear();
	for (int idx = 0; idx < input_size; idx++) {
		batch_input.push_back(inputs[idx]);
		if (batch_input.size() == m_options.batchSize || idx == input_size - 1) {
			batch_outputs.clear();
			m_driver.decode(batch_input, batch_outputs);
			batch_input.clear();
			outputs.insert(outputs.end(), batch_outputs.begin(), batch_outputs.end());
		}
	}
}


int main(int argc, char* argv[]) {
	std::string trainFile = "", devFile = "", testFile = "", modelFile = "./model.bin";
	std::string optionFile = "";
	std::string outputFile = "";
	bool bTrain = false;
	dsr::Argument_helper ah;
	int threads = 1;

	ah.new_flag("l", "learn", "train or test", bTrain);
	ah.new_named_string("train", "trainCorpus", "named_string", "training corpus to train a model, must when training", trainFile);
	ah.new_named_string("dev", "devCorpus", "named_string", "development corpus to train a model, optional when training", devFile);
	ah.new_named_string("test", "testCorpus", "named_string",
		"testing corpus to train a model or input file to test a model, optional when training and must when testing", testFile);
	ah.new_named_string("model", "modelFile", "named_string", "model file, must when training and testing", modelFile);
	ah.new_named_string("option", "optionFile", "named_string", "option file to train a model, optional when training", optionFile);
	ah.new_named_string("output", "outputFile", "named_string", "output file to test, must when testing", outputFile);
	ah.new_named_int("th", "thread", "named_int", "number of threads for openmp", threads);

	ah.process(argc, argv);

	DisParser parser;
	if (bTrain) {
		parser.train(trainFile, devFile, testFile, modelFile, optionFile);
	}
	else {
		parser.test(testFile, outputFile, modelFile, optionFile);
	}

	return 0;
}
