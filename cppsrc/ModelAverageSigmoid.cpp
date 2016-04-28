#include "ModelAverageSigmoid.h"

float ModelAverageSigmoid::estSetRating(int user, std::vector<int> items, 
    Eigen::VectorXf& sumItemFac) {
  float r_us_est = 0;

  sumItemFac.fill(0);
  for (auto&& item: items) {
    r_us_est += estItemRating(user, item);
    sumItemFac += V.row(item);
  }
  r_us_est = r_us_est/items.size();

  float dev = r_us_est - u_m[user];
  r_us_est = sigmoid(dev, g_k);

  return r_us_est;
}


float ModelAverageSigmoid::estSetRating(int user, std::vector<int>& items) {
  float r_us_est = 0;
  
  for (auto&& item: items) {
    r_us_est += estItemRating(user, item);
  }
  r_us_est = r_us_est/items.size();

  float dev = r_us_est - u_m[user];
  r_us_est = sigmoid(dev, g_k);

  return r_us_est;
}


void ModelAverageSigmoid::train(const Data& data, const Params& params, 
    Model& bestModel) {
  
  std::cout << "ModelAverage::train" << std::endl;

  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  std::cout << "Train RMSE: " << rmse(data.trainSets) << std::endl;
 
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
 
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  std::vector<int> items; 
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
      items = uSet.itemSets[setInd].first;
      float r_us = uSet.itemSets[setInd].second;

      //estimate rating on the set and latent factor sum
      float r_us_est = 0;

      sumItemFactors.fill(0);
      for (auto&& item: items) {
        r_us_est += estItemRating(user, item);
        sumItemFactors += V.row(item);
      }
      r_us_est = r_us_est/items.size();

      float dev = r_us_est - u_m[user];
      r_us_est = sigmoid(dev, g_k);

      float commGradCoeff = exp(-1.0*g_k*dev)*r_us_est*r_us_est;
      commGradCoeff = 2*(r_us_est - r_us)*commGradCoeff;

      //user gradient
      grad = ((g_k*commGradCoeff)/items.size())*sumItemFactors;
      
      //update user
      U.row(user) -= learnRate*(grad.transpose() + 2.0*uReg*U.row(user));

      //update items
      grad = ((g_k*commGradCoeff)/items.size())*U.row(user);
      for (auto&& item: items) {
        V.row(item) -= learnRate*(grad.transpose() + 2.0*iReg*V.row(item));
      }

      //u_m gradient
      float u_mGrad = commGradCoeff*-1.0*g_k + 2.0*u_m[user]*u_mReg;
      //update u_m
      u_m[user] -= learnRate*u_mGrad;
      
      //compute g_k grad
      float g_kGrad = commGradCoeff*dev + 2.0*g_k*g_kReg;
      //update g_k
      g_k -= learnRate*g_kGrad;
    }
    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj)) {
        break;
      }
      std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
        << prevValRMSE << " best val RMSE:" << bestValRMSE 
        << " train RMSE:" << rmse(data.trainSets) << std::endl;
    }

  }

}

