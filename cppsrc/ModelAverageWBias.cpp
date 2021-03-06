#include "ModelAverageWBias.h"

float ModelAverageWBias::estItemRating(int user, int item) {
  bool uFound = false, iFound = true;
  float rating = 0;
  if (trainUsers.find(user) != trainUsers.end() && 
      invalidUsers.find(user) == invalidUsers.end()) {
    //found in train and not in invalid
    uFound = true;
    rating += uBias(user);
  }
  if (trainItems.find(item) != trainItems.end()) {
    iFound = true;
    rating += iBias(item);
  }
  if (uFound && iFound) {
    rating += U.row(user).dot(V.row(item));
  }
  return rating;
}


float ModelAverageWBias::estSetRating(int user, std::vector<int>& items) {
  int setSz = items.size();
  float r_us = 0;
  for (auto&& item: items) {
    r_us += estItemRating(user, item);
  }
  r_us = r_us/setSz;
  return r_us;
}


float ModelAverageWBias::objective(const std::vector<UserSets>& uSets, 
    gk_csr_t *mat) {
  return Model::objective(uSets, mat);
}


float ModelAverageWBias::objective(const std::vector<UserSets>& uSets) {
  return Model::objective(uSets);
}


void ModelAverageWBias::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelAverageWBias::train" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
  int nTrUsers = (int)uInds.size(); 

  //initialize global bias
  int nTrainSets = 0;
  float meanSetRating = 0;
  /*
  for (auto&& uInd: uInds) {
    const auto& uSet = data.trainSets[uInd];
    for (auto&& itemSet: uSet.itemSets) {
      meanSetRating += itemSet.second;
      nTrainSets++;
    } 
  }
  */
  //meanSetRating = meanSetRating/nTrainSets;
  //gBias = meanSetRating;

  trainUsers = data.trainUsers;
  trainItems = data.trainItems;
  std::cout << "train Users: " << trainUsers.size() 
    << " trainItems: " << trainItems.size() << std::endl;
  
  auto partUIRatingsTup = getUIRatingsTup(data.partTrainMat);

  //initialize random engine
  std::mt19937 mt(params.seed);
  std::uniform_int_distribution<int> dist(0, 1000);

  auto uSetInds = getUserSetInds(data.trainSets);
  if (params.isMixRat) {
    std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
    updateFacBiasUsingRatMat(partUIRatingsTup);
  }

  for (int iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uSetInds.begin(), uSetInds.end(), mt);
    for (const auto& uSetInd: uSetInds) {
      int uInd = uSetInd.first;
      int setInd = uSetInd.second;
      const UserSets& uSet = data.trainSets[uInd];
      int user = uSet.user;
            
      if (uSet.itemSets.size() == 0) {
        std::cerr << "!! zero size user itemset found !! " << user << std::endl; 
        continue;
      }
      auto items = uSet.itemSets[setInd].first;
      float r_us = uSet.itemSets[setInd].second;

      if (items.size() == 0) {
        std::cerr << "!! zero size itemset found !!" << std::endl; 
        continue;
      }

      //estimate rating on the set
      float r_us_est = estSetRating(user, items);

      //compute sum of item latent factors
      sumItemFactors.fill(0);    
      for (auto item: items) {
        sumItemFactors += V.row(item);
      }

      //user gradient
      grad = (2.0*(r_us_est - r_us)/items.size())*sumItemFactors
              + 2.0*uReg*U.row(user).transpose();

      //update user
      U.row(user) -= learnRate*(grad.transpose());
      
      //update user bias
      uBias(user) -= learnRate*((2.0*(r_us_est - r_us)) + 2.0*uBiasReg*uBias(user));

      //update items
      grad = (2.0*(r_us_est - r_us)/items.size())*U.row(user);
      for (auto&& item: items) {
        tempGrad = grad + 2.0*iReg*V.row(item).transpose(); 
        V.row(item) -= learnRate*(tempGrad.transpose());
        //update item bias
        iBias(item) -= learnRate*((2.0*(r_us_est - r_us)/items.size()) 
            + 2.0*iBiasReg*iBias(item));
      }

    }   

    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacBiasUsingRatMat(partUIRatingsTup);
    }

    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE)) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      
      if (iter%100 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << std::endl;
      }

    }

  }
  
  /*
  float pickyRMSE = 0, nonPickyRMSE = 0;
  int pickyUCount = 0, nonPickyUCount = 0;
  std::unordered_set<int> pickyU, nonPickyU;
  for (const auto& userSets : data.trainSets) {
    auto p_u = userSets.getVarPickiness(data.ratMat);
    if (p_u[2] != -99) {
      if (fabs(p_u[2]) > 0.5) {
        pickyU.insert(userSets.user);
      } else {
        nonPickyU.insert(userSets.user);
      }
    }
  }
  
  std::cout << "picky users: " << pickyU.size() << " nonPicky users: " 
    << nonPickyU.size() << std::endl;
  std::cout << "pickyU Set RMSE: " << rmse(data.testSets, pickyU) << std::endl;
  std::cout << "Non-pickyU Set RMSE: " << rmse(data.testSets, nonPickyU) << std::endl;
  std::cout << "pickyU item RMSE: " << rmseNotSets(data.allSets, data.ratMat, 
      data.partTrainMat, pickyU) << std::endl;
  std::cout << "Non-pickyU itme RMSE: " << rmseNotSets(data.allSets, data.ratMat, 
      data.partTrainMat, nonPickyU) << std::endl;
  */

}

