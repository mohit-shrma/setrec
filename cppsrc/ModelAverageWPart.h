#ifndef _MODEL_AVERAGE_WPART_H_
#define _MODEL_AVERAGE_WPART_H_

#include "ModelAverageWBias.h"

class ModelAverageWPart: public ModelAverageWBias {

  public:
    ModelAverageWPart(const Params& params):ModelAverageWBias(params) {}
    virtual void train(const Data& data, const Params& params, Model& bestModel);
};

#endif
