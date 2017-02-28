#ifndef _MODEL_H_
#define _MODEL_H_

#include <Eigen/Dense>
#include <random>
#include <tuple>
#include <cmath>

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
    
    //user diversity biases
    Eigen::VectorXf uDivWt;

    //user memb wts to function
    Eigen::MatrixXf UWts;
    int nWts;

    //sigmoid steepness
    float g_k;

    //global set bias
    float gBias;
    
    //size of latent factors
    int facDim;
    
    //regularization
    float uReg, iReg;
    float uSetBiasReg; //needed for variance based model
    float uBiasReg, iBiasReg, gBiasReg;
    float gamma; //nneded for varaince based model

    float learnRate;

    std::vector<float> globalItemRatings;
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
    bool isTerminateModelWPartIRMSE(Model& bestModel, 
        const Data& data, int iter, int& bestIter, float& bestObj, float& prevObj, 
        float& bestValRMSE, float& prevValRMSE);
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
    bool isTerminateRankSetModel(Model& bestModel, const Data& data, int iter, 
      int& bestIter, float& prevRecall, float& bestValRecall) ;
    float rmse(const std::vector<UserSets>& uSets);
    float rmse(const UserSets& uSet);
    float rmse(const std::vector<UserSets>& uSets, std::unordered_set<int>& valUsers);
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
  std::pair<std::vector<float>, std::vector<float>> precisionNCall(
    const std::vector<UserSets>& uSets, gk_csr_t *mat, std::vector<int>& Ns, 
    float ratingThresh);  
  float fracCorrOrderedSets(const std::vector<UserSets>& uSets);
  float fracCorrOrderedSets(const std::vector<UserSets>& uSets, float lb);
  float precisionN(gk_csr_t* testMat, gk_csr_t* valMat, gk_csr_t* trainMat,
    int N);
  float corrOrderedItems(
    std::vector<std::vector<std::pair<int, float>>> testRatings);
  float corrOrderedItems(
    std::vector<std::vector<std::pair<int, float>>> testRatings, float lb);
  float rmseNotSets(const std::vector<UserSets>& uSets, gk_csr_t *mat);
  float rmseNotSets(const std::vector<UserSets>& uSets, gk_csr_t *mat, 
      gk_csr_t *partTrainMat);
  float rmseNotSets(const std::vector<UserSets>& uSets, gk_csr_t *mat, 
      gk_csr_t *partTrainMat, std::unordered_set<int>& validUsers);
  std::pair<float, float> fracCorrOrderedRatingsUser(int user, 
    std::vector<std::pair<int, float>> itemRatings);
  float matCorrOrderedRatingsWOSets(const std::vector<UserSets>& uSets, 
      gk_csr_t *mat);
  std::pair<float, float> fracCorrOrderedRatingsUserTop(int user, 
    std::vector<std::pair<int, float>> itemRatings, float lb);
  float matCorrOrderedRatingsWOSetsTop(
    const std::vector<UserSets>& uSets, gk_csr_t *mat, float lb);
  float corrOrderedItems(gk_csr_t *mat, float lb);
  float computeEntropy(int user, ItemsSet& item);
  void updateFacUsingRatMat(std::vector<std::tuple<int, int, float>>& ratings); 
  void updateFacBiasUsingRatMat(std::vector<std::tuple<int, int, float>>& ratings);

};


#endif
