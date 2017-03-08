#ifndef _MODEL_WT_AVG_ALL_RANGE_H_
#define _MODEL_WT_AVG_ALL_RANGE_H_

#include "Model.h"
#include "alglib/stdafx.h"
#include "alglib/optimization.h"

class ModelWtAverageAllRange: public Model {
  public:
    ModelWtAverageAllRange(const Params& params):Model(params) {}
    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items);
    void estSetRatings(int user, const std::vector<int>& items,
        std::vector<float>& setRatings);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    void trainQP(const Data& data, const Params& params, Model& bestModel);
    void trainQPSmooth(const Data& data, const Params& params, Model& bestModel);
    void trainGreedy(const Data& data, const Params& params, Model& bestModel);
    float estSetRating(int user, const std::vector<int>& items, int exSetInd);
    float estUExSetRMSE(const UserSets& uSet, int exSetInd);
    float estUSetsRMSE(const UserSets& uSet, alglib::real_1d_array& wts);
    float estUSetsRMSE(Eigen::MatrixXf& Q, Eigen::VectorXf& c, 
        alglib::real_1d_array& wts);
};


#endif

