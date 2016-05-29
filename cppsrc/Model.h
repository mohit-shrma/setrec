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
 
    //user-item bias
    Eigen::VectorXf uBias;
    Eigen::VectorXf iBias;
    
    //user set biases
    Eigen::VectorXf uSetBias;
    
    //global set bias
    float gBias;
    
    //size of latent factors
    int facDim;
    
    //regularization
    float uReg, iReg, uSetBiasReg;
    float uBiasReg, iBiasReg;
    float gamma;

    float learnRate;

    std::map<int, float> globalItemRatings;
    std::unordered_set<int> trainUsers, trainItems;
    std::unordered_set<int> invalidUsers;
    
    Model(const Params &params);
    Model(const Params &params, const char* uFacName, const char* iFacName);

    virtual float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items) {
      //std::cerr << "Base class: estSetRating not defined" << std::endl;
      return 0;
    }
    virtual float objective(const std::vector<UserSets>& uSets);
    virtual float objective(const std::vector<UserSets>& uSets, gk_csr_t *mat);
    virtual float objective(gk_csr_t *mat);
    virtual void train(const Data& data, const Params& params, Model& bestModel) {
      std::cerr << "Base class: train not defined" << std::endl;
    }
    
    bool isTerminateModel(Model& bestModel, const Data& data, int iter, 
        int& bestIter, float& bestObj, float& prevObj, float& bestValRMSE,
        float& prevValRMSE); 
    bool isTerminateModelWPart(Model& bestModel, const Data& data, int iter, 
        int& bestIter, float& bestObj, float& prevObj, float& bestValRMSE,
        float& prevValRMSE); 
    bool isTerminateModelWPart(Model& bestModel, const Data& data, int iter, 
        int& bestIter, float& bestObj, float& prevObj); 
    bool isTerminateModel(Model& bestModel, const Data& data, int iter, 
        int& bestIter, float& bestObj, float& prevObj); 
    bool isTerminateRecallModel(Model& bestModel, const Data& data, int iter,
      int& bestIter, float& bestRecall, float& prevRecall, float& bestValRecall,
      float& prevValRecall);
    float rmse(const std::vector<UserSets>& uSets);
    float rmse(const std::vector<UserSets>& uSets, gk_csr_t *mat);
    std::map<int, float> itemRMSE(const std::vector<UserSets>& uSets,
      gk_csr_t *mat);
    float rmse(gk_csr_t *mat);
    float rmse(gk_csr_t *mat, std::unordered_set<int>& valItems);
    float spearmanRankN(gk_csr_t *mat, int N);
    float spearmanRankN(gk_csr_t *mat, const std::vector<UserSets>& uSets, 
        int N);
    float inversionCount(gk_csr_t *mat, const std::vector<UserSets>& uSets, 
      int N);
    float invertRandPairCount(gk_csr_t *mat, const std::vector<UserSets>& uSets,
        int seed);
    std::string modelSign();
    void save(std::string opPrefix);
    void load(std::string opPrefix);
    float recallTopN(gk_csr_t *mat, const std::vector<UserSets>& uSets,
      int N);
    float recallTopN(gk_csr_t *mat, const std::vector<UserSets>& uSets,
      std::unordered_set<int>& invalUsers, int N);
    float recallHit(const std::vector<UserSets>& uSets,
      std::map<int, int> uItems, 
      std::map<int, std::unordered_set<int>> ignoreUItems, int N);
    float ratingsNDCG(
      std::map<int, std::map<int, float>> uRatings);
    float ratingsNDCGRel(
      std::map<int, std::map<int, float>> uRatings);
    std::pair<float, float> ratingsNDCGPrecK(const std::vector<UserSets>& uSets,
        std::map<int, std::map<int, float>> uRatings,
        int N);
  float ratingsNDCGRelRand(
      std::map<int, std::map<int, float>> uRatings,
      std::mt19937& mt);
  float invertRandPairCount(
    std::vector<std::tuple<int, int, int>> allTriplets);
  std::pair<float, float> precisionNCall(const std::vector<UserSets>& uSets, 
      gk_csr_t *mat, int N, float ratingThresh);  
};


#endif
