#ifndef _MODEL_AVERAGE_BPR_W_BIAS_TOP_H_
#define _MODEL_AVERAGE_BPR_W_BIAS_TOP_H_

#include "ModelAverageBPRWBias.h"

class ModelAverageBPRWBiasTop: public ModelAverageBPRWBias {
  public:
    ModelAverageBPRWBiasTop(const Params& params)
      :ModelAverageBPRWBias(params){}
    virtual void train(const Data& data, const Params& params, 
        Model& bestModel);
    virtual bool isTerminateRankSetModel(Model& bestModel, 
      const Data& data, int iter, int& bestIter, float& prevValRecall,
      float& bestValRecall, float lb);
};


#endif

