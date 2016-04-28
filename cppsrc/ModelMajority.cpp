#include "ModelMajority.h"

float ModelMajority::estSetRating(int user, std::vector<int>& items, 
    Eigen::VectorXf& sumFac) {
  
  float majSz = std::ceil(((float)items.size()) / 2); 
  float majRat = 0;
  
  //get item ratings in decreasing order
  std::vector<std::pair<int, float>> itemRatings;
  for (auto&& item: items) {
    itemRatings.push_back(std::make_pair(item, estItemRating(user, item)));
  }
  std::sort(itemRatings.begin(), itemRatings.end(), descComp);

  sumFac.fill(0);
  for (int i = 0; i < majSz; i++) {
    majRat += itemRatings[i].second;
    sumFac += V.row(itemRatings[i].first);
  }
  
  majRat = majRat/majSz;
  
  return majRat;
}


float ModelMajority::estSetRating(int user, std::vector<int>& items) {
  
  float majSz = std::ceil(((float)items.size()) / 2); 
  float majRat = 0;

  //get item ratings in decreasing order
  std::vector<std::pair<int, float>> itemRatings;
  for (auto&& item: items) {
    itemRatings.push_back(std::make_pair(item, estItemRating(user, item)));
  }
  std::sort(itemRatings.begin(), itemRatings.end(), descComp);

  for (int i = 0; i < majSz; i++) {
    majRat += itemRatings[i].second;
  }
  
  majRat = majRat/majSz;
  
  return majRat;
}


void ModelMajority::train(const Data& data, const Params& params, Model& bestModel) {
  
  std::cout << "ModelMajority::train" << std::endl;

  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  std::cout << "Train RMSE: " << rmse(data.trainSets) << std::endl;
 
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter;
  std::vector<std::pair<int, float>> itemRatings;
  std::vector<int> items;
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
      items = uSet.itemSets[setInd].first;
      float r_us = uSet.itemSets[setInd].second;
      float majSz = std::ceil(((float)items.size()) / 2); 
      float r_us_est = 0;
      
      itemRatings.clear();
      for (auto&& item: items) {
        itemRatings.push_back(std::make_pair(item, estItemRating(user, item)));
      }
      std::sort(itemRatings.begin(), itemRatings.end(), descComp);
     
      sumItemFactors.fill(0);
      for (int i = 0; i < majSz; i++) {
        r_us_est += itemRatings[i].second;
        sumItemFactors += V.row(itemRatings[i].first);
      }
      r_us_est = r_us_est/majSz;
      
      //user gradient
      grad = (2.0*(r_us_est - r_us)/majSz)*sumItemFactors;
      
      //update user
      U.row(user) -= learnRate*(grad.transpose() + 2.0*uReg*U.row(user));

      //update items
      grad = (2.0*(r_us_est - r_us)/majSz)*U.row(user);
      for (int i = 0; i < majSz; i++) {
        int item = itemRatings[i].first;
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
        << " train RMSE:" << rmse(data.trainSets) << std::endl;
    }

  }

}



