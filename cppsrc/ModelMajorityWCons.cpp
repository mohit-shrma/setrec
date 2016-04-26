#include "ModelMajorityWCons.h"

std::pair<float, float> ModelMajorityWCons::setRatingNMaxRat(int user,
    std::vector<int>& items) {
  
  float majSz = std::ceil(((float)items.size()) / 2); 
  float r_us_est = 0;
  
  //get item ratings in decreasing order
  std::vector<std::pair<int, float>> itemRatings;
  for (auto&& item: items) {
    itemRatings.push_back(std::make_pair(item, estItemRating(user, item)));
  }
  std::sort(itemRatings.begin(), itemRatings.end(), descComp);

  for (int i = 0; i < majSz; i++) {
    r_us_est += itemRatings[i].second;
  }

  r_us_est = r_us_est/majSz;
  
  return std::make_pair(r_us_est, itemRatings[0].second);
}


float ModelMajorityWCons::objective(const std::vector<UserSets>& uSets) {
  
  float obj = 0.0, uRegErr = 0.0, iRegErr = 0.0;
  float norm, setScore, diff, maxRat;
  int user, nSets = 0, constViolCt = 0;
  
  for (auto&& uSet: uSets) {
    user = uSet.user;
    for (size_t i = 0; i < uSet.itemSets.size(); i++) {
      auto items = uSet.itemSets[i];
      auto setNMax = setRatingNMaxRat(user, items);
      setScore = setNMax.first;
      maxRat = setNMax.second;
      diff = setScore - uSet.setScores[i];
      obj += diff*diff;
      if (uSet.setScores[i] > maxRat) {
        //constraint violated
        obj += constWt*(uSet.setScores[i] - maxRat);
        constViolCt++;
      }
      nSets++;
    }
  }
  
  norm = U.norm();
  uRegErr = uReg*norm*norm;

  norm = V.norm();
  iRegErr = iReg*norm*norm;

  obj += uRegErr + iRegErr;

  return obj;

  

}


void ModelMajorityWCons::train(const Data& data, const Params& params, 
    Model& bestModel) {
  
  std::cout << "ModelMajorityWCons::train" << std::endl;

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
      items = uSet.itemSets[setInd];
      float r_us = uSet.setScores[setInd];
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
      if (itemRatings[0].second < r_us) {
        //constraint violated
        grad += -1.0*constWt*V.row(itemRatings[0].first);
      }

      //update user
      U.row(user) -= learnRate*(grad.transpose() + 2.0*uReg*U.row(user));

      //update items except the max
      grad = (2.0*(r_us_est - r_us)/majSz)*U.row(user);
      for (int i = 1; i < majSz; i++) {
        int item = itemRatings[i].first;
        V.row(item) -= learnRate*(grad.transpose() + 2.0*iReg*V.row(item));
      }
      //update maxItem
      if (itemRatings[0].second < r_us) {
        //constraint violated
        grad += -1*constWt*U.row(user);
      }
      V.row(itemRatings[0].first) -= learnRate*(grad.transpose() 
          + 2.0*iReg*V.row(itemRatings[0].first));
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

