#ifndef _MODEL_AVERAGE_WGBIAS_PART_H_
#define _MODEL_AVERAGE_WGBIAS_PART_H_

#include "svdFrmsvdlib.h"
#include "ModelAverageWGBias.h"

class ModelAverageGBiasWPart:public ModelAverageWGBias {
  public:
    ModelAverageGBiasWPart(const Params& params):ModelAverageWGBias(params){}
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets, gk_csr_t *mat);
    virtual bool isTerminateModelWPartIRMSE(Model& bestModel, 
      const Data& data, int iter, int& bestIter, float& bestObj, float& prevObj, 
      float& bestValRMSE, float& prevValRMSE);
};


#endif

