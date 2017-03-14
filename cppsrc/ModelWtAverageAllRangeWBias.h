#ifndef _MODEL_WT_AVG_ALLRANGE_W_BIAS_H_
#define _MODEL_WT_AVG_ALLRANGE_W_BIAS_H_

#include "ModelWtAverageAllRange.h"

class ModelWtAverageAllRangeWBias: public ModelWtAverageAllRange {
  public:
    ModelWtAverageAllRangeWBias(const Params& params):ModelWtAverageAllRange(params) {}
    virtual float estItemRating(int user, int item);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    void trainQPSmooth(const Data& data, const Params& params, Model& bestModel);
};

#endif
