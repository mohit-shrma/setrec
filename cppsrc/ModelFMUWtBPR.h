#ifndef _MODEL_FM_U_WT_BPR_H_
#define _MODEL_FM_U_WT_BPR_H_

#include "ModelAverageBPRWBiasTop.h"

class ModelFMUWtBPR: public ModelAverageBPRWBiasTop {
  public:
    ModelFMUWtBPR(const Params& params): ModelAverageBPRWBiasTop(params) {}
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float estSetRating(int user, std::vector<int>& items);
    float estSetRating(int user, std::vector<int>& items, 
      Eigen::VectorXf& sumItemFactors, float& avgItemsPairwiseSim);
};

#endif
