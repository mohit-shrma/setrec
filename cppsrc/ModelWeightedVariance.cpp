#include "ModelWeightedVariance.h"


float ModelWeightedVariance::estItemRating(int user, int item) {
  bool uFound = false, iFound = true;
  float rating = 0;
  if (trainUsers.find(user) != trainUsers.end() && 
      invalidUsers.find(user) == invalidUsers.end()) {
    //found in train and not in invalid
    uFound = true;
    //rating += uBias(user);
  }
  if (trainItems.find(item) != trainItems.end()) {
    iFound = true;
    //rating += iBias(item);
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
  std::cout << " uDovWt norm: " << norm << std::endl;
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
  //gBias = meanSetRating;

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
        const UserSets& uSet = data.trainSets[uInd];
        int user = uSet.user;
              
        if (uSet.itemSets.size() == 0) {
          std::cerr << "!! zero size user itemset found !! " << user << std::endl; 
          continue;
        }
        //select a set at random
        int setInd = dist(mt) % uSet.itemSets.size();
        auto items = uSet.itemSets[setInd].first;
        const float r_us = uSet.itemSets[setInd].second;
        const int setSz = items.size();

        if (setSz == 0) {
          std::cerr << "!! zero size itemset found !!" << std::endl; 
          continue;
        }
          
        //estimate rating on the set
        float r_us_est = 0; //TODO:estSetRating(user, items);
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

        if (var > EPS) {
          stdDev = std::sqrt(var);
          varCoeff = 0.5/stdDev;
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
        U.row(user) -= learnRate*(grad.transpose());
        
        //update items
        grad = (2.0*(r_us_est - r_us)/setSz)*U.row(user);
        float temp;
        for (auto&& item: items) {
          temp = 1;
          temp += uDivWt(user)*varCoeff * 2 * tempFac.dot(V.row(item));
          temp -= ((uDivWt(user)*varCoeff)/setSz) * 2 * ratSum;
          tempGrad = temp*grad + 2.0*iReg*V.row(item).transpose(); 
          V.row(item) -= learnRate*(tempGrad.transpose());
        }
        
        //update user weight
        uDivWt(user) -= learnRate*(2.0*(r_us_est - r_us)*(gamma + stdDev) 
          + 2.0*uSetBiasReg*uDivWt(user));
      }
    }    
  
    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacUsingRatMat(partUIRatingsTup);
    }
    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      /*
      if ((!params.isMixRat && isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE))) {
        break;
        //save best model
        //bestModel.save(params.prefix);
      } else if ((params.isMixRat && isTerminateModelWPartIRMSE(bestModel, data, iter, 
            bestIter, bestObj, prevObj, bestValRMSE, prevValRMSE))) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }*/
      if (isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE)) {
        break;
      }

      if (iter % 100 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << " U norm: " << U.norm() << " V norm: " << V.norm() 
          << " uDivWt norm: " << uDivWt.norm()
          << std::endl;
      }

    }

  }
  
  /*
  std::ofstream opFile("User_var_weights.txt");
  for (int u = 0; u < nUsers; u++) {
    if (trainUsers.find(u) != trainUsers.end()) {
      opFile << u << " " << uDivWt(u) << std::endl;
    }
  }
  opFile.close();
  */
}


