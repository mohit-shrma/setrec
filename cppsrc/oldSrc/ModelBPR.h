#ifndef _MODEL_BPR_H_
#define _MODEL_BPR_H_

#include "Model.h"


class ModelBPR: public Model {
  public:
    ModelBPR(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    bool isTerminatePrecisionModel(Model& bestModel, const Data& data,
      int iter, int& bestIter, float& bestValRecall, float& prevValRecall);
    bool isTerminatePrecisionModel(Model& bestModel, const Data& data,
      std::vector<std::vector<std::pair<int, float>>> testRatings,
      int iter, int& bestIter, float& bestValRecall, float& prevValRecall);
};

#endif

