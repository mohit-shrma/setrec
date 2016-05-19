#ifndef _MODEL_AVERAGE_SETBIAS_W_PART_H_
#define _MODEL_AVERAGE_SETBIAS_W_PART_H_

#include "ModelAverageWPart.h"

class ModelAverageSetBiasWPart: public ModelAverageWPart {
  public:
    ModelAverageSetBiasWPart(const Params& params):ModelAverageWPart(params){}
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual float objective(const std::vector<UserSets>& uSets, gk_csr_t *mat);
};

#endif

