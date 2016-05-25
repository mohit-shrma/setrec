#include "ModelAverageWSetBias.h"


float ModelAverageWSetBias::estSetRating(int user, std::vector<int>& items) {
  int setSz = items.size();
  float r_us = 0;
  for (auto&& item: items) {
    r_us += estItemRating(user, item);
  }
  r_us = r_us/setSz;
  //add user's set rating bias
  r_us += uSetBias(user);
  return r_us;
}


float ModelAverageWSetBias::objective(const std::vector<UserSets>& uSets) {
  float obj = ModelAverageWBias::objective(uSets);
  float norm = 0;
  
  //add user biases regularization
  norm = uSetBias.norm();
  obj += norm*norm*uReg;

  return obj;
}


void ModelAverageWSetBias::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelAverageWSetBias::train" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  std::cout << "Train RMSE: " << rmse(data.trainSets) << std::endl;
  
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  Eigen::MatrixXf uGradsAcc(nUsers, facDim);
  uGradsAcc.fill(0);
  Eigen::MatrixXf iGradsAcc(nItems, facDim);
  iGradsAcc.fill(0);
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
  int nTrUsers = (int)uInds.size(); 

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

        //update user set bias
        uSetBias(user) -= learnRate*(2.0*(r_us_est - r_us) 
            + 2.0*uSetBiasReg*uSetBias(user));

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
          << std::endl;
      }

      bestModel.save(params.prefix);
    }

  }



}

