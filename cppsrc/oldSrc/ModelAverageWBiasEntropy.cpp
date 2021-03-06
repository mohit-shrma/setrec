#include "ModelAverageWBiasEntropy.h"


float ModelAverageWBiasEntropy::estSetRating(int user, ItemsSet& items) {
  int setSz = items.size();
  float r_us = 0;
  for (auto&& item: items) {
    r_us += estItemRating(user, item);
  }
  r_us = r_us/setSz;
  r_us += uDivWt(user)*computeEntropy(user, items);
  return r_us;
}


float ModelAverageWBiasEntropy::objective(const std::vector<UserSets>& uSets, 
    gk_csr_t *mat) {
  float obj = ModelAverageWBias::objective(uSets, mat);
  for (auto&& user: trainUsers) {
    obj += uSetBiasReg*uDivWt(user)*uDivWt(user);
  }
  return obj;
}


float ModelAverageWBiasEntropy::objective(const std::vector<UserSets>& uSets) {
  float obj = ModelAverageWBias::objective(uSets);
  for (auto&& user: trainUsers) {
    obj += uSetBiasReg*uDivWt(user)*uDivWt(user);
  }
  return obj;
}


void ModelAverageWBiasEntropy::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelAverageWBiasEntropy::train" << std::endl; 
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
  for (auto&& uInd: uInds) {
    const auto& uSet = data.trainSets[uInd];
    for (auto&& itemSet: uSet.itemSets) {
      meanSetRating += itemSet.second;
      nTrainSets++;
    } 
  }
  meanSetRating = meanSetRating/nTrainSets;
  gBias = meanSetRating;

  trainUsers = data.trainUsers;
  trainItems = data.trainItems;
  std::cout << "train Users: " << trainUsers.size() 
    << " trainItems: " << trainItems.size() << std::endl;

  //initialize random engine
  std::mt19937 mt(params.seed);
  std::uniform_int_distribution<int> dist(0, 1000);

  for (int iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uInds.begin(), uInds.end(), mt);
    for (int i = 0; i < data.nTrainSets/nTrUsers; i++) {
      for (auto&& uInd: uInds) {
        const UserSets& uSet = data.trainSets[uInd];
        int user = uSet.user;
              
        if (uSet.itemSets.size() == 0) {
          std::cerr << "!! zero size user itemset found !! " << user << std::endl; 
          continue;
        }
        //select a set at random
        int setInd = dist(mt) % uSet.itemSets.size();
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

        //update user-specific entropy weight
        float entropy = computeEntropy(user, items); 
        uDivWt(user) -= learnRate*(2.0*(r_us_est - r_us)*entropy + 2.0*uSetBiasReg*uDivWt(user)); 
      }
    }

    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE)) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      
      if (iter%10 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << std::endl;
      }

    }

  }

}


