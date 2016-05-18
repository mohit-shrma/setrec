#include "ModelAverageBiasesOnly.h"

float ModelAverageBiasesOnly::estItemRating(int user, int item) {
  return uBias(user) + iBias(item);
} 


float ModelAverageBiasesOnly::estSetRating(int user, std::vector<int>& items) {
  
 float ratSum = 0;

 for (auto&& item: items) {
  ratSum += estItemRating(user, item);
 }
 
 ratSum = ratSum/items.size();

 ratSum += uSetBias(user);
 ratSum += gBias;

 return ratSum;
}


float ModelAverageBiasesOnly::objective(const std::vector<UserSets>& uSets) {
  
  float obj = 0.0;
  float norm, setScore, diff;
  int user, nSets = 0;
  
  for (auto&& uSet: uSets) {
    user = uSet.user;
    for (size_t i = 0; i < uSet.itemSets.size(); i++) {
      auto items = uSet.itemSets[i].first;
      setScore = estSetRating(user, items);
      diff = setScore - uSet.itemSets[i].second;
      obj += diff*diff;
      nSets++;
    }
  }  

  norm = uBias.norm();
  obj += uBiasReg*norm*norm;

  norm = iBias.norm();
  obj += iBiasReg*norm*norm;

  norm = uSetBias.norm();
  obj += norm*norm*uSetBiasReg; 

  return obj;
}


void ModelAverageBiasesOnly::train(const Data& data, const Params& params,
    Model& bestModel) {

  std::cout << "ModelAverageBiasesOnly::train" << std::endl; 
  
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter, nTrSets  = 0;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
  int nTrUsers = (int)uInds.size(); 

  //compute global bias as global mean
  for (auto&& uSet: data.trainSets) {
    for (auto&& itemSet: uSet.itemSets) {
      gBias += itemSet.second; 
      nTrSets++;
    }
  }
  gBias = gBias/nTrSets;
  
  std::cout << "gBias: " << gBias << " nTrSets: " << nTrSets << std::endl;
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  std::cout << "Train RMSE: " << rmse(data.trainSets) << std::endl;
  
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
          std::cerr << "!! zero size user itemset foundi !! " << user << std::endl; 
          continue;
        }
        //select a set at random
        int setInd = dist(mt) % uSet.itemSets.size();
        auto items = uSet.itemSets[setInd].first;
        float r_us = uSet.itemSets[setInd].second;

        if (items.size() == 0) {
          std::cerr << "!! zero size itemset foundi !!" << std::endl; 
          continue;
        }

        //estimate rating on the set
        float r_us_est = estSetRating(user, items);
        
        //update user bias
        uBias(user) -= learnRate*((2.0*(r_us_est - r_us)) + 2.0*uBiasReg*uBias(user));

        //update user set bias
        uSetBias(user) -= learnRate*(2.0*(r_us_est - r_us) 
            + 2.0*uSetBiasReg*uSetBias(user));

        //update items
        for (auto&& item: items) {
          //update item bias
          iBias(item) -= learnRate*((2.0*(r_us_est - r_us)/items.size()) 
              + 2.0*iBiasReg*iBias(item));
        }

      }
    }   

    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE)) {
        //save best model
        bestModel.save(params.prefix);
        break;
      }
 
      
      if (iter % 10 == 0) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << " recall@10: " << recallTopN(data.ratMat, data.trainSets, 10)
          << " spearman@10: " << spearmanRankN(data.ratMat, data.trainSets, 10)
          << " invCount@10: " << inversionCount(data.partTestMat, data.trainSets, 10)
          << std::endl;
      }

      //bestModel.save(params.prefix);
    }

  }

}


