#ifndef _MODEL_AVERAGE_H_
#define _MODEL_AVERAGE_H_

#include "Model.h"

class ModelAverage: public Model {
  
  public:
    ModelAverage(const Params& params):Model(params) {}  
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
};


#endif



