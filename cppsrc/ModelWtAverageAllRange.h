#ifndef _MODEL_WT_AVG_ALL_RANGE_H_
#define _MODEL_WT_AVG_ALL_RANGE_H_

#include "Model.h"

class ModelWtAverageAllRange: public Model {
  public:
    ModelWtAverageAllRange(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    float estSetRating(int user, std::vector<int>& items, 
        std::vector<Eigen::VectorXf>& cumSumItemFactors, 
        std::vector<std::pair<int, float>>& setItemRatings,
        std::vector<float>& cumSumPreds);
};


#endif

