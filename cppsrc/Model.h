#ifndef _MODEL_H_
#define _MODEL_H_

#include <Eigen/Dense>
#include <random>
#include <tuple>

#include "datastruct.h"
#include "const.h"
#include "mathUtil.h"
#include "util.h"

class Model {
  
  public:
    int nUsers;
    int nItems;
   
    //user-item latent factors
    Eigen::MatrixXf U;
    Eigen::MatrixXf V;
    
    //size of latent factors
    int facDim;
    
    //regularization
    float uReg, iReg;

    float learnRate;

    Model(const Params &params);
    
    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items) {
      std::cerr << "Base class: estSetRating not defined" << std::endl;
      return -1;
    }
    virtual float objective(const std::vector<UserSets>& uSets);
    virtual void train(const Data& data, const Params& params, Model& bestModel) {
      std::cerr << "Base class: train not defined" << std::endl;
    }
    
    bool isTerminateModel(Model& bestModel, const Data& data, int iter, 
        int& bestIter, float& bestObj, float& prevObj, float& bestValRMSE,
        float& prevValRMSE); 
    bool isTerminateModel(Model& bestModel, const Data& data, int iter, 
        int& bestIter, float& bestObj, float& prevObj); 
    float rmse(const std::vector<UserSets>& uSets);
    float rmse(const std::vector<UserSets>& uSets, gk_csr_t *mat);
    float rmse(gk_csr_t *mat);
    float spearmanRankN(gk_csr_t *mat, int N);
    float spearmanRankN(gk_csr_t *mat, const std::vector<UserSets>& uSets, 
        int N);
    std::string modelSign();
    void save(std::string opPrefix);
    void load(std::string opPrefix);
    float recallTopN(gk_csr_t *mat, const std::vector<UserSets>& uSets,
      int N);
};


#endif
