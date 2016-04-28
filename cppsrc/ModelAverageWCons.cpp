#include "ModelAverageWCons.h"


std::pair<float, float> ModelAverageWCons::setRatingNMaxRat(
    int user, std::vector<int>& items) {
  float maxRat = 0, r_us_est = 0;
  int maxItem = -1;
  for (auto&& item: items) {
    float r_ui = estItemRating(user, item);
    r_us_est += r_ui;
    if (r_ui > maxRat || -1 == maxItem) {
      maxItem = item;
      maxRat = r_ui;
    }
  }
  r_us_est = r_us_est/items.size();
  return std::make_pair(r_us_est, maxRat);
}


float ModelAverageWCons::objective(const std::vector<UserSets>& uSets) {
  
  float obj = 0.0, uRegErr = 0.0, iRegErr = 0.0;
  float norm, setScore, diff, maxRat;
  int user, nSets = 0, constViolCt = 0;
  
  for (auto&& uSet: uSets) {
    user = uSet.user;
    for (size_t i = 0; i < uSet.itemSets.size(); i++) {
      auto items = uSet.itemSets[i].first;
      auto setNMax = setRatingNMaxRat(user, items);
      setScore = setNMax.first;
      maxRat = setNMax.second;
      diff = setScore - uSet.itemSets[i].second;
      obj += diff*diff;
      if (uSet.itemSets[i].second > maxRat) {
        //constraint violated
        obj += constWt*(uSet.itemSets[i].second - maxRat);
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


void ModelAverageWCons::train(const Data& data, const Params& params, 
    Model& bestModel) {
  
  std::cout << "ModelAverageConst::train" << std::endl;

  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  std::cout << "Train RMSE: " << rmse(data.trainSets) << std::endl;
 
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
 
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempgrad(facDim);
  
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
      auto items = uSet.itemSets[setInd].first;
      float r_us = uSet.itemSets[setInd].second;
      int maxRatItem = -1;
      float maxRat = -1;
      float r_us_est = 0, r_ui_est;
      
      //estimate rating on the set, sum item latent factor and get max rating
      sumItemFactors.fill(0);
      for (auto item: items) {
        sumItemFactors += V.row(item);
        r_ui_est = estItemRating(user, item);
        r_us_est += r_ui_est;
        if (r_ui_est > maxRat || maxRatItem == -1) {
          maxRatItem = item;
          maxRat = r_ui_est;
        }
      }
      r_us_est = r_us_est/items.size();

      //user gradient
      grad = (2.0*(r_us_est - r_us)/items.size())*sumItemFactors;
      if (maxRat < r_us) {
        //if constraint violated
        grad += -1.0*constWt*V.row(maxRatItem);
      }

      //update user
      U.row(user) -= learnRate*(grad.transpose() + 2.0*uReg*U.row(user));

      //update items
      tempgrad = (2.0*(r_us_est - r_us)/items.size())*U.row(user);
      for (auto&& item: items) {
        grad = tempgrad;
        if (item == maxRatItem && maxRat < r_us) {
          //constraint violated
          grad += -1*constWt*U.row(user);
        }
        //update item
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



