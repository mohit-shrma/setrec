#ifndef _MODEL_WT_AVERAGE_H_
#define _MODEL_WT_AVERAGE_H_

#include "ModelAverageWBias.h"
#include "stdafx.h"
#include "optimization.h"


class ModelWtAverage: public ModelAverageWBias {
  public:
    ModelWtAverage(const Params& params):ModelAverageWBias (params) {}
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    void estSetRatings(int user, const std::vector<int>& items, 
        float& r_us1, float& r_us2, float& r_us3);
};

#endif



