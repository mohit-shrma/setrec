#ifndef _MODEL_AVERAGE_SIGMOID_H_
#define _MODEL_AVERAGE_SIGMOID_H_

#include "Model.h"
#include "mathUtil.h"

class ModelAverageSigmoid: public Model {
  
  public:
    std::vector<float> u_m; //user shift param
    float u_mReg;           //user shift reg
    float g_k;              //global steepness param
    float g_kReg;           //global steepness reg
    ModelAverageSigmoid(const Params& params):Model(params) {
      u_mReg = params.u_mReg;
      g_kReg = params.g_kReg;
      //random engine
      std::mt19937 mt(params.seed);
      std::uniform_real_distribution<> dis(0, 1);
      //initialize user shift param
      for (int i = 0; i < params.nUsers; i++) {
        u_m.push_back(dis(mt));
      }

      //initialize global steepness
      g_k = dis(mt);
    }

    virtual float estSetRating(int user, std::vector<int>& items);
    virtual void train(const Data& data, const Params& params, Model& bestModel);
    float estSetRating(int user, std::vector<int> items, Eigen::VectorXf& sumItemFac);
};

#endif
