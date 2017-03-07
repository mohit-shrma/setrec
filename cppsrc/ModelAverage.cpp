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
  
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  Eigen::MatrixXf uGradsAcc(nUsers, facDim);
  uGradsAcc.fill(0);
  Eigen::MatrixXf iGradsAcc(nItems, facDim);
  iGradsAcc.fill(0);
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter;
  
  Eigen::MatrixXf UGradSqAvg(nUsers, facDim);
  Eigen::MatrixXf VGradSqAvg(nItems, facDim);
  UGradSqAvg.fill(0); VGradSqAvg.fill(0);
 
  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
  int nTrUsers = (int)uInds.size(); 

  trainUsers = data.trainUsers;
  trainItems = data.trainItems;
  std::cout << "train Users: " << trainUsers.size() 
    << " trainItems: " << trainItems.size() << std::endl;
  
  auto partUIRatingsTup = getUIRatingsTup(data.partTrainMat);
  
  //initialize random engine
  std::mt19937 mt(params.seed);
  std::uniform_int_distribution<int> dist(0, 1000);

  for (int iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uInds.begin(), uInds.end(), mt);
    for (int i = 0; i < data.nTrainSets/nTrUsers; i++) {
      for (auto&& uInd: uInds) {
        UserSets uSet = data.trainSets[uInd];
        int user = uSet.user;
              
        if (uSet.itemSets.size() == 0) {
          std::cerr << "!! zero size user itemset foundi !! " << user << std::endl; 
          continue;
        }
        //select a set at random
        int setInd = dist(mt) % uSet.itemSets.size();
        auto items = uSet.itemSets[setInd].first;
        float r_us = uSet.itemSets[setInd].second;
        int setSz = items.size();

        if (setSz == 0) {
          std::cerr << "!! zero size itemset foundi !!" << std::endl; 
          continue;
        }
  
        //estimate rating on the set
        float r_us_est = 0;

        //compute sum of item latent factors
        sumItemFactors.fill(0);    
        for (auto item: items) {
          sumItemFactors += V.row(item);
        }
        r_us_est += (U.row(user).dot(sumItemFactors)) / setSz;

        //user gradient
        grad = (2.0*(r_us_est - r_us)/items.size())*sumItemFactors
                + 2.0*uReg*U.row(user).transpose();
        //update user
        //U.row(user) -= learnRate*(grad.transpose());
        RMSPropUpdate(U, user, UGradSqAvg, grad, learnRate, 0.9);

        //update items
        grad = (2.0*(r_us_est - r_us)/items.size())*U.row(user);
        for (auto&& item: items) {
          tempGrad = grad + 2.0*iReg*V.row(item).transpose(); 
          //V.row(item) -= learnRate*(tempGrad.transpose());
          RMSPropUpdate(V, item, VGradSqAvg, tempGrad, learnRate, 0.9);
        }
      }
    }

    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg);
    }

    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      
      if ((!params.isMixRat && isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE, false))) {
        break;
        //save best model
        //bestModel.save(params.prefix);
      } else if ((params.isMixRat && isTerminateModelWPartIRMSE(bestModel, data, iter, 
            bestIter, bestObj, prevObj, bestValRMSE, prevValRMSE))) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      
      /*
      if (isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE)) {
        break;
      }
      */
      if (iter % 100 == 0  || iter == params.maxIter-1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << " bIter: " << bestIter
          << std::endl;
      }
    }

  }

}





