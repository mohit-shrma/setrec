#include "ModelWeightedVarianceWBias.h"

float ModelWeightedVarianceWBias::estItemRating(int user, int item) {
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


void ModelWeightedVarianceWBias::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelWeightedVarianceWBias::train" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf ratWtSumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  Eigen::VectorXf tempFac(facDim);
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter;
  
  Eigen::VectorXf uDivWtSqAvg(nUsers);
  Eigen::MatrixXf UGradSqAvg(nUsers, facDim);
  Eigen::MatrixXf VGradSqAvg(nItems, facDim);
  Eigen::VectorXf uBiasGradSqAvg(nUsers);
  Eigen::VectorXf iBiasGradSqAvg(nItems);
  UGradSqAvg.fill(0); VGradSqAvg.fill(0);
  uBiasGradSqAvg.fill(0); iBiasGradSqAvg.fill(0); uDivWtSqAvg.fill(0); 

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
  int nTrUsers = (int)uInds.size(); 

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
    updateFacBiasUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg, 
        uBiasGradSqAvg, iBiasGradSqAvg);
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
      const float r_us = uSet.itemSets[setInd].second;
      const int setSz = items.size();

      if (setSz == 0) {
        std::cerr << "!! zero size itemset found !!" << std::endl; 
        continue;
      }
        
      //estimate rating on the set
      float r_us_est = 0;
      float ratSum = 0;
      float ratSqrSum = 0;
      float rating = 0;
      float mean, var, stdDev = EPS, varCoeff = 0;
      //compute sum of item latent factors
      sumItemFactors.fill(0);   
      ratWtSumItemFactors.fill(0);
      for (auto item: items) {
        rating = U.row(user).dot(V.row(item)) + uBias(user) + iBias(item);
        ratSum += rating;
        ratSqrSum += rating*rating;
        sumItemFactors += V.row(item);
        ratWtSumItemFactors += rating*V.row(item);
      }
      mean = ratSum/setSz;
      var = ratSqrSum/setSz - mean*mean;
      r_us_est = mean;

      if (var > EPS) {
        stdDev = std::sqrt(var); //TODO: investigate sqrt(var +  gamma)
        varCoeff = 0.5/stdDev;
      }
      r_us_est += uDivWt(user)*(gamma + stdDev); //TODO: investigate sqrt(gamma + var)

      //user gradient
      grad = (sumItemFactors/setSz) 
        + ((uDivWt(user)*varCoeff/setSz) * 2.0 * ratWtSumItemFactors) 
        - ((uDivWt(user)*varCoeff/(setSz*setSz)) * 2.0 * ratSum * sumItemFactors); 
      grad *= 2.0*(r_us_est - r_us); 
      grad += 2.0*uReg*U.row(user).transpose();
      
      tempFac = U.row(user);

      //update user
      //U.row(user) -= learnRate*(grad.transpose());
      RMSPropUpdate(U, user, UGradSqAvg, grad, learnRate, 0.9);
      
      //update user bias
      float biasGrad = 2.0*(r_us_est - r_us) + 2.0*uBiasReg*uBias(user); 
      uBiasGradSqAvg(user) = 0.9*uBiasGradSqAvg(user) + 0.1*biasGrad*biasGrad;
      uBias(user) -= (learnRate/std::sqrt(uBiasGradSqAvg(user) + 1e-8))*biasGrad;
 
      //update user weight
      float uDivWtGrad = 2.0*(r_us_est - r_us)*(gamma + stdDev) 
        + 2.0*uSetBiasReg*uDivWt(user);
      //uDivWt(user) -= learnRate*(uDivWtGrad);
      uDivWtSqAvg(user) = 0.9*uDivWtSqAvg(user) + 0.1*uDivWtGrad*uDivWtGrad;
      uDivWt(user) -= (learnRate/(std::sqrt(uDivWtSqAvg(user) + 1e-8)))*uDivWtGrad;

     
      //update items
      grad = (2.0*(r_us_est - r_us)/setSz)*U.row(user);
      float temp;
      for (auto&& item: items) {
        temp = 1;
        temp += uDivWt(user)*varCoeff * 2 * tempFac.dot(V.row(item));
        temp -= ((uDivWt(user)*varCoeff)/setSz) * 2 * ratSum;
        tempGrad = temp*grad + 2.0*iReg*V.row(item).transpose(); 
        //V.row(item) -= learnRate*(tempGrad.transpose());
         RMSPropUpdate(V, item, VGradSqAvg, tempGrad, learnRate, 0.9);
      }
      
      //update item bias
      for (auto&& item: items) {
        temp = 2*uDivWt(user)*varCoeff/setSz;
        temp = temp*((uBias(user) + iBias(item) + U.row(user).dot(V.row(item))) - mean);
        temp += 1.0/setSz;
        biasGrad = 2.0*(r_us_est - r_us)*temp + 2.0*iBiasReg*iBias(item);
        iBiasGradSqAvg(item) = 0.9*iBiasGradSqAvg(item) + 0.1*biasGrad*biasGrad;
        iBias(item) -= (learnRate/std::sqrt(iBiasGradSqAvg(item) + 1e-8))*biasGrad; 
      }

    }    
  
    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacBiasUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg, 
          uBiasGradSqAvg, iBiasGradSqAvg);
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

      if (iter % 250 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val: " 
          << prevValRMSE << " best val:" << bestValRMSE 
          << " train:" << rmse(data.trainSets) 
          << " train ratings: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings: " << rmse(data.testSets, data.ratMat)
          << " U norm: " << U.norm() << " V norm: " << V.norm() 
          << " uDivWt norm: " << uDivWt.norm()
          << " bestIter: " << bestIter
          << std::endl;
      }

    }

  }
   
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
  std::cout << "Non-pickyU item RMSE: " << rmseNotSets(data.allSets, data.ratMat, 
      data.partTrainMat, nonPickyU) << std::endl;

  
  /*
  std::ofstream opFile("User_var_weights.txt");
  std::cout << "No trainUsers: " << trainUsers.size() 
    << " no trainSets: " << data.trainSets.size() << std::endl;
  for (const auto& userSets : data.trainSets) {
    auto p_u = userSets.getVarPickiness(data.ratMat);
    if (p_u[2] != -99.0) {
      opFile << userSets.user << " " << uDivWt(userSets.user) << " " << p_u[0] << " " << p_u[1] << " " << p_u[2] << std::endl;
    }
  }
  opFile.close();
  */   
}



