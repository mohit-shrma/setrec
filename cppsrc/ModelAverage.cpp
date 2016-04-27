#include "ModelAverage.h"


float ModelAverage::estSetRating(int user, std::vector<int>& items) {
  int setSz = items.size();
  float ratSum = 0;
  
  for (auto&& item: items) {
    ratSum += estItemRating(user, item);
  }

  return ratSum/setSz;
}


void ModelAverage::train(const Data& data, const Params& params, Model& bestModel) {
  
  std::cout << "ModelAverage::train" << std::endl;

  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  std::cout << "Train RMSE: " << rmse(data.trainSets) << std::endl;
 
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
 
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  
  //initialize random engine
  std::mt19937 mt(params.seed);
  std::uniform_int_distribution<int> dist(0, 1000);

  for (int iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uInds.begin(), uInds.end(), mt);
    for (auto&& uInd: uInds) {
      UserSets uSet = data.trainSets[uInd];
      int user = uSet.user;
            
      //select a set at random
      int setInd = dist(mt) % uSet.itemSets.size();
      auto items = uSet.itemSets[setInd];
      float r_us = uSet.setScores[setInd];

      //estimate rating on the set
      float r_us_est = estSetRating(user, items);
      
      //compute sum of item latent factors
      sumItemFactors.fill(0);    
      for (auto item: items) {
        sumItemFactors += V.row(item);
      }

      //user gradient
      grad = (2.0*(r_us_est - r_us)/items.size())*sumItemFactors;
      
      //update user
      U.row(user) -= learnRate*(grad.transpose() + 2.0*uReg*U.row(user));

      //update items
      grad = (2.0*(r_us_est - r_us)/items.size())*U.row(user);
      for (auto&& item: items) {
        V.row(item) -= learnRate*(grad.transpose() + 2.0*iReg*V.row(item));
      }
    }
    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj)) {
        break;
      }
      std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
        << prevValRMSE << " best val RMSE:" << bestValRMSE 
        << " train RMSE:" << rmse(data.trainSets) 
        << " ratings RMSE: " << rmse(data.ratMat) << std::endl;
    }

  }

}





