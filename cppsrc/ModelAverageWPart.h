#ifndef _MODEL_AVERAGE_WPART_H_
#define _MODEL_AVERAGE_WPART_H_

#include "ModelAverageWBias.h"
#include "svdFrmsvdlib.h"

class ModelAverageWPart: public ModelAverageWBias {

  public:
    ModelAverageWPart(const Params& params):ModelAverageWBias(params) {}
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    virtual float objective(const std::vector<UserSets>& uSets, gk_csr_t *mat);
    void trainIter(const Data& data, const Params& params, Model& bestModel);
    void trainJoint(const Data& data, const Params& params, Model& bestModel);
    bool isTerminateModelWPartIRMSE(Model& bestModel, 
      const Data& data, int iter, int& bestIter, float& bestObj, float& prevObj, 
      float& bestValRMSE, float& prevValRMSE);
};

#endif
