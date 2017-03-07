#include "ModelFM.h"

float ModelFM::estSetRating(int user, std::vector<int>& items) {
  
  float r_us = 0;
  float fmSqrSum = 0;
  float fmSumsqr = 0;
  
  r_us += gBias;
  r_us += uBias(user);

  //add item biases
  for (auto&& item: items) {
    r_us += iBias(item);
  }

  for (int k = 0; k < facDim; k++) {
    float tempSqrsum = 0;
    
    //go over items
    for (auto&& item: items) {
      tempSqrsum += V.row(item)[k];
      fmSumsqr += V.row(item)[k] * V.row(item)[k];
    } 
    
    //user
    tempSqrsum += U.row(user)[k];
    tempSqrsum = tempSqrsum*tempSqrsum;
    fmSqrSum += tempSqrsum;
    fmSumsqr += U.row(user)[k] * U.row(user)[k];
  }
  
  r_us += 0.5*(fmSqrSum - fmSumsqr);

  return r_us;
}


float ModelFM::estSetRating(int user, std::vector<int>& items, 
    Eigen::VectorXf& sumFactors) {
  
  float r_us = 0;
  float fmSqrSum = 0;
  float fmSumsqr = 0;
  
  sumFactors.fill(0);
  r_us += gBias;
  r_us += uBias(user);

  //add item biases
  for (auto&& item: items) {
    r_us += iBias(item);
  }

  for (int k = 0; k < facDim; k++) {
    float tempSqrsum = 0;
    
    //go over items
    for (auto&& item: items) {
      tempSqrsum += V.row(item)[k];
      fmSumsqr += V.row(item)[k] * V.row(item)[k];
      sumFactors[k] += V.row(item)[k];
    } 
    
    //user
    tempSqrsum += U.row(user)[k];
    tempSqrsum = tempSqrsum*tempSqrsum;
    fmSqrSum += tempSqrsum;
    fmSumsqr += U.row(user)[k] * U.row(user)[k];
    sumFactors[k] += U.row(user)[k];
  }
  
  r_us += 0.5*(fmSqrSum - fmSumsqr);

  return r_us;
}


float ModelFM::estItemRating(int user, int item) {
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


float ModelFM::objective(const std::vector<UserSets>& uSets, 
    gk_csr_t *mat) {
  return Model::objective(uSets, mat) + gBiasReg*gBias*gBias;
}


float ModelFM::objective(const std::vector<UserSets>& uSets) {
  return Model::objective(uSets) + gBiasReg*gBias*gBias;
}


void ModelFM::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelFM::train" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  
  Eigen::VectorXf sumFactors(facDim);
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
  //gBias = meanSetRating;

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

        //estimate rating on the set & sum of item factors
        float r_us_est = estSetRating(user, items, sumFactors);

        //user gradient
        grad = sumFactors - U.row(user).transpose();
        grad = (2.0*(r_us_est - r_us))*(grad)
                + 2.0*uReg*U.row(user).transpose();

        //update user
        U.row(user) -= learnRate*(grad.transpose());
        
        //update user bias
        uBias(user) -= learnRate*((2.0*(r_us_est - r_us)) + 2.0*uBiasReg*uBias(user));

        //update items
        for (auto&& item: items) {
          grad = sumFactors - V.row(item).transpose();
          grad = (2.0*(r_us_est - r_us))*(grad);
          tempGrad = grad + 2.0*iReg*V.row(item).transpose(); 
          V.row(item) -= learnRate*(tempGrad.transpose());
          //update item bias
          iBias(item) -= learnRate*((2.0*(r_us_est - r_us)) 
              + 2.0*iBiasReg*iBias(item));
        }
        
        //update global bias
        gBias -= learnRate*(2.0*(r_us_est - r_us)) + 2.0*gBiasReg*gBias;
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
          << std::endl;
      }

    }

  }
  std::cout << "gBias: " << gBias << std::endl;
}


