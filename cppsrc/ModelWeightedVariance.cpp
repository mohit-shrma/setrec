#include "ModelWeightedVariance.h"


float ModelWeightedVariance::estItemRating(int user, int item) {
  bool uFound = false, iFound = true;
  float rating = 0;
  if (trainUsers.find(user) != trainUsers.end() && 
      invalidUsers.find(user) == invalidUsers.end()) {
    //found in train and not in invalid
    uFound = true;
  }
  if (trainItems.find(item) != trainItems.end()) {
    iFound = true;
  }
  if (uFound && iFound) {
    rating += U.row(user).dot(V.row(item));
  }
  return rating;
}


float ModelWeightedVariance::estSetRating(int user, std::vector<int>& items) {
 
  float r_us = 0; 

  int setSz = items.size();

  std::vector<float> preds(setSz, 0);
  //get predictions
  float ratSum = 0;
  float ratSqrSum = 0;
  float rating;
  for (int i = 0; i < setSz; i++) {
    int item = items[i];
    rating = estItemRating(user, item);
    ratSum += rating;
    ratSqrSum += rating*rating;
  }

  float mean = ratSum/setSz;
  float var = (ratSqrSum/setSz) - (mean*mean);
  float std = EPS;
  if (var > EPS) {
    std = std::sqrt(var);
  }
  r_us = mean + uDivWt(user)*(std + gamma); 

  return r_us;
}


float ModelWeightedVariance::objective(const std::vector<UserSets>& uSets, 
    gk_csr_t *mat) {
  float norm = uDivWt.norm();
  float obj = Model::objective(uSets, mat);
  obj += norm*norm*uSetBiasReg;
  //std::cout << " uDovWt norm: " << norm << std::endl;
  return obj;
}


float ModelWeightedVariance::objective(const std::vector<UserSets>& uSets) {
  float obj =  Model::objective(uSets);
  float norm = uDivWt.norm();
  obj += norm*norm*uSetBiasReg;
  return obj;
}


void ModelWeightedVariance::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelWeightedVariance::train" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf ratWtSumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  Eigen::VectorXf tempFac(facDim);
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  int bestIter;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
  int nTrUsers = (int)uInds.size(); 
 
  Eigen::VectorXf uDivWtSqAvg(nUsers);
  Eigen::MatrixXf UGradSqAvg(nUsers, facDim);
  Eigen::MatrixXf VGradSqAvg(nItems, facDim);
  UGradSqAvg.fill(0); VGradSqAvg.fill(0); uDivWtSqAvg.fill(0);

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
    updateFacUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg);
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
      //select a set at random
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
        rating = U.row(user).dot(V.row(item));
        ratSum += rating;
        ratSqrSum += rating*rating;
        sumItemFactors += V.row(item);
        ratWtSumItemFactors += rating*V.row(item);
      }
      mean = ratSum/setSz;
      var = ratSqrSum/setSz - mean*mean;
      r_us_est = mean;

      //stdDev = std::sqrt(var + gamma);
      //varCoeff = 0.5/stdDev;

      if (var > EPS) {
        stdDev = std::sqrt(var);
        varCoeff = 0.5/stdDev;
        //varCoeff = 0.5/(stdDev + gamma);
      }
      r_us_est += uDivWt(user)*(gamma + stdDev);

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
      
      //update user weight
      //uDivWt(user) -= learnRate*(2.0*(r_us_est - r_us)*(stdDev) 
      float uDivWtGrad = 2.0*(r_us_est - r_us)*(gamma + stdDev) 
        + 2.0*uSetBiasReg*uDivWt(user);
      //uDivWt(user) -= learnRate*(uDivWtGrad);
      uDivWtSqAvg(user) = 0.9*uDivWtSqAvg(user) + 0.1*uDivWtGrad*uDivWtGrad;
      uDivWt(user) -= (learnRate/(std::sqrt(uDivWtSqAvg(user) + 1e-8)))*uDivWtGrad;
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
       
      
      //if (isTerminateModel(bestModel, data, iter, bestIter,
      //      bestObj, prevObj, bestValRMSE, prevValRMSE)) {
      //  break;
      //}
      

      if (iter % 200 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << " U norm: " << U.norm() << " V norm: " << V.norm() 
          << " uDivWt norm: " << uDivWt.norm()
          << " bestIter: " << bestIter
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
  
  std::cout << "pickyU Set RMSE: " << rmse(data.testSets, pickyU) << std::endl;
  std::cout << "Non-pickyU Set RMSE: " << rmse(data.testSets, nonPickyU) << std::endl;
  std::cout << "pickyU item RMSE: " << rmseNotSets(data.allSets, data.ratMat, 
      data.partTrainMat, pickyU) << std::endl;
  std::cout << "Non-pickyU itme RMSE: " << rmseNotSets(data.allSets, data.ratMat, 
      data.partTrainMat, nonPickyU) << std::endl;
  */
  /*
  std::ofstream opFile("User_var_weights.txt");
  std::cout << std::endl;
  std::cout << "No trainUsers: " << trainUsers.size() 
    << " no trainSets: " << data.trainSets.size() << std::endl;
  float avgDiff = 0, avgEstSetRMSE = 0, avgFitRMSE = 0, count  = 0;
  for (const auto& userSets : data.trainSets) {
    auto p_u = userSets.getVarPickiness(data.ratMat);
    float estSetsRMSE = rmse(userSets);
    if (p_u[2] != -99.0) {
      float fitRMSE = p_u[3];
      float diff = fabs(estSetsRMSE - fitRMSE);
      avgDiff += diff;
      avgEstSetRMSE += estSetsRMSE;
      avgFitRMSE += fitRMSE;
      count += 1;
      opFile << userSets.user << " " << uDivWt(userSets.user) << " " << p_u[0] << " " << p_u[1] << " " << p_u[2] << " " << diff << std::endl;
    }
  }
  opFile.close();
  std::cout << "avg Diff: " << avgDiff/count << " avgFitRMSE: " 
    << avgFitRMSE/count << " avgEstSetRMSE: " << avgEstSetRMSE/count 
    << " " << count << std::endl;
  */
}


