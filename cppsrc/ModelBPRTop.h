#ifndef _MODEL_BPR_TOP_H_
#define _MODEL_BPR_TOP_H_

#include "ModelBPR.h"

class ModelBPRTop: public ModelBPR {
  public:
    ModelBPRTop(const Params& params):ModelBPR(params){}
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    bool isTerminatePrecisionModel(Model& bestModel, const Data& data,
      std::vector<std::vector<std::pair<int, float>>> testRatings,
      int iter, int& bestIter, float& bestValRecall, float& prevValRecall) {
};

#endif

