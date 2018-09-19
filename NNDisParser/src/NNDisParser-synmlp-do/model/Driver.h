#ifndef Driver_H_
#define Driver_H_

#include "N3LDG.h"
#include "HyperParams.h"
#include "ModelParams.h"
#include "GreedyGraph.h"
#include "BeamGraph.h"
#include "Action.h"

class Driver {
public:
	Driver() {}

	~Driver() {}

public:
	GreedyGraphBuilder _greedyBuilder;
	BeamGraphBuilder _beamBuilder;

	HyperParams _hyperparams;
	ModelParams _modelparams;
	Graph _encoderGraph;
	Graph _decoderGraph;
	ModelUpdate _ada;
	ModelUpdate _beam_ada;
	Metric _eval;
	bool _useBeam;

	inline void initial() {
		if (!_hyperparams.bValid()) {
			std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
			return;
		}
		if (!_modelparams.initial(_hyperparams)) {
			std::cout << "model parameter initialization Error, Please check!" << std::endl;
			return;
		}
		_greedyBuilder.initial(_modelparams, _hyperparams);

		dtype dropout_value = _hyperparams.dropProb;
		_hyperparams.dropProb = -1.0; // don't drop out in beam graph nodes.
		_beamBuilder.initial(_modelparams, _hyperparams);

		_hyperparams.dropProb = dropout_value;

		setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
		_hyperparams.print();
		_useBeam = false;
	}

public:
	inline void setUpdateParameters(const dtype &nnRegular, const dtype &adaAlpha, const dtype &adaEps) {
		_ada._alpha = adaAlpha;
		_ada._eps = adaEps;
		_ada._reg = nnRegular;

		_beam_ada._alpha = adaAlpha * 0.1;
		_beam_ada._eps = adaEps;
		_beam_ada._reg = nnRegular;
	}

	inline void setDropFactor(dtype drop_factor) {
		_encoderGraph.setDropFactor(drop_factor);
		_decoderGraph.setDropFactor(drop_factor);
	}



	inline dtype train(const vector<Instance> &documents) {
		_eval.reset();
		dtype cost = 0.0;
		int num = documents.size();

		if (!_useBeam) {
			for (int idx = 0; idx < num; idx++) {
				_encoderGraph.clearValue(true);
				const Instance &doc = documents[idx];
				_greedyBuilder.encode(&_encoderGraph, doc);

				_encoderGraph.compute();

				_decoderGraph.clearValue(true);
				_greedyBuilder.decode(&_decoderGraph, doc, &doc.gold_actions);
				cost += loss_google(_greedyBuilder, num);
				_decoderGraph.backward();
				_eval.overall_label_count += doc.gold_actions.size();
				_encoderGraph.backward();
			}
		}
		else {

			for (int idx = 0; idx < num; idx++) {
				_encoderGraph.clearValue(true);
				const Instance &doc = documents[idx];
				_beamBuilder.encode(&_encoderGraph, doc);
				_encoderGraph.compute();

				_decoderGraph.clearValue(true);

				_beamBuilder.decode(&_decoderGraph, doc, &doc.gold_actions);
				cost += loss_google(_beamBuilder, num);
				_decoderGraph.backward();
				//_eval.overall_label_count += gold_actions[idx].size();
				_encoderGraph.backward();
			}
		}
		return cost;
	}

	inline dtype loss_google(const GreedyGraphBuilder &builder, int batch) {
		int maxstep = builder.outputs.size();
		if (maxstep == 0) return 1.0;
		//_eval.correct_label_count += maxstep;
		PNode pBestNode = NULL;
		PNode pGoldNode = NULL;
		PNode pCurNode;
		dtype sum, max;
		int curcount, goldIndex;
		vector<dtype> scores;
		dtype cost = 0.0;

		for (int step = 0; step < maxstep; step++) {
			curcount = builder.outputs[step].size();
			if (curcount == 1) {
				_eval.correct_label_count++;
				continue;
			}
			max = 0.0;
			goldIndex = -1;
			pBestNode = pGoldNode = NULL;
			for (int idx = 0; idx < curcount; idx++) {
				pCurNode = builder.outputs[step][idx].in;
				if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
					pBestNode = pCurNode;
				}
				if (builder.outputs[step][idx].is_gold) {
					pGoldNode = pCurNode;
					goldIndex = idx;
				}
			}

			if (goldIndex == -1) {
				std::cout << "impossible" << std::endl;
			}
			pGoldNode->loss[0] = -1.0 / batch;

			max = pBestNode->val[0];
			sum = 0.0;
			scores.resize(curcount);
			for (int idx = 0; idx < curcount; idx++) {
				pCurNode = builder.outputs[step][idx].in;
				scores[idx] = exp(pCurNode->val[0] - max);
				sum += scores[idx];
			}

			for (int idx = 0; idx < curcount; idx++) {
				pCurNode = builder.outputs[step][idx].in;
				pCurNode->loss[0] += scores[idx] / (sum * batch);
			}

			if (pBestNode == pGoldNode)_eval.correct_label_count++;
			//_eval.overall_label_count++;

			cost += -log(scores[goldIndex] / sum);

			if (std::isnan(cost)) {
				std::cout << "std::isnan(cost), google loss,  debug" << std::endl;
			}

		}

		return cost;
	}

	dtype loss_google(const BeamGraphBuilder &builder, int batch) {
		int maxstep = builder.outputs.size();
		if (maxstep == 0) return 1.0;
		//_eval.correct_label_count += maxstep;
		PNode pBestNode = NULL;
		PNode pGoldNode = NULL;
		PNode pCurNode;
		dtype sum, max;
		int curcount, goldIndex;
		vector<dtype> scores;
		dtype cost = 0.0;

		for (int step = maxstep - 1; step < maxstep; step++) { // TRY STEP 0
			curcount = builder.outputs[step].size();
			_eval.overall_label_count++;
			if (curcount == 1) {
				_eval.correct_label_count++;
				continue;
			}
			max = 0.0;
			goldIndex = -1;
			pBestNode = pGoldNode = NULL;
			for (int idx = 0; idx < curcount; idx++) {
				pCurNode = builder.outputs[step][idx].in;
				if (pBestNode == NULL || pCurNode->val[0] > pBestNode->val[0]) {
					pBestNode = pCurNode;
				}
				if (builder.outputs[step][idx].is_gold) {
					pGoldNode = pCurNode;
					goldIndex = idx;
				}
			}

			if (goldIndex == -1) {
				std::cout << "impossible" << std::endl;
			}
			/*
			pGoldNode->loss[0] = -1.0 / batch;

			max = pBestNode->val[0];
			sum = 0.0;
			scores.resize(curcount);
			for (int idx = 0; idx < curcount; idx++) {
				pCurNode = builder.outputs[step][idx].in;
				scores[idx] = exp(pCurNode->val[0] - max);
				sum += scores[idx];
			}

			for (int idx = 0; idx < curcount; idx++) {
				pCurNode = builder.outputs[step][idx].in;
				pCurNode->loss[0] += scores[idx] / (sum * batch);
			}

			*/

			if (pGoldNode != pBestNode) {
				pGoldNode->loss[0] = -1.0 / batch;
				pBestNode->loss[0] = 1.0 / batch;

				cost += 1.0;
			}
			if (pBestNode == pGoldNode)_eval.correct_label_count++;
			//_eval.overall_label_count++;

			//cost += -log(scores[goldIndex] / sum);

			if (std::isnan(cost)) {
				std::cout << "std::isnan(cost), google loss,  debug" << std::endl;
			}

		}

		return cost;
	}

	inline void decode(const vector<Instance> &documents, vector<CResult> &results) {
		int step, num = documents.size();
		if (!_useBeam) {
			results.resize(num);
			for (int idx = 0; idx < num; idx++) {
				_encoderGraph.clearValue();
				const Instance &doc = documents[idx];
				_greedyBuilder.encode(&_encoderGraph, doc);

				_encoderGraph.compute();

				_decoderGraph.clearValue();
				_greedyBuilder.decode(&_decoderGraph, doc);
				step = _greedyBuilder.outputs.size();
				_greedyBuilder.states[step - 1].getResults(results[idx], _hyperparams);
			}
		}
		else {
			results.resize(num);
			for (int idx = 0; idx < num; idx++) {
				_encoderGraph.clearValue();
				const Instance &doc = documents[idx];
				_beamBuilder.encode(&_encoderGraph, doc);

				_encoderGraph.compute();

				_decoderGraph.clearValue();
				_beamBuilder.decode(&_decoderGraph, doc);
				step = _beamBuilder.outputs.size();
				_beamBuilder.states[step - 1][0].getResults(results[idx], _hyperparams);
			}
		}

	}

	inline void updateModel() {
		if (!_useBeam) {
			if (_ada._params.empty()) {
				_modelparams.exportModelParams(_ada);
			}
			_ada.updateAdam(_hyperparams.clips);
		}
		else {
			if (_beam_ada._params.empty()) {
				_modelparams.exportModelBeamParams(_beam_ada);
			}
			_beam_ada.updateAdam(_hyperparams.clips);
		}
	}

	inline void setGraph(bool useBeam) {
		_useBeam = useBeam;
	}

};

#endif /* Driver_H_ */
