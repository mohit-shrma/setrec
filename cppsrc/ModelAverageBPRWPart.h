#ifndef _MODEL_AVERAGE_BPR_W_PART_
#define _MODEL_AVERAGE_BPR_W_PART_H_

#include "ModelAverageBPRWBiasTop.h"

class ModelAverageBPRWPart: public ModelAverageBPRWBiasTop {
  public:
    ModelAverageBPRWPart(const Params& params):ModelAverageBPRWBiasTop(params) {}
    bool isTerminatePrecisionModel(Model& bestModel, const Data& data,
        int iter, int& bestIter, float& bestValRecall, float& prevValRecall);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
};

#endif

