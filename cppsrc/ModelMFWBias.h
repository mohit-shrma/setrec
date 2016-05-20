#ifndef _MODEL_MF_W_BIAS_H_
#define _MODEL_MF_W_BIAS_H_

#include "Model.h"
#include "svdFrmsvdlib.h"

class ModelMFWBias: public Model {
  public:
    ModelMFWBias(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item); 
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    float isTerminateModelIRMSE(Model& bestModel, const Data& data,
      int iter, int& bestIter, float& bestObj, float& prevObj, float& bestValRMSE,
      float& prevValRMSE);
};


#endif
