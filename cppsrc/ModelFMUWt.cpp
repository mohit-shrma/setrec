#include "ModelFMUWt.h"


float ModelFMUWt::estSetRating(int user, std::vector<int>& items) {
 
  float r_us = 0;
  float fmSqrSum = 0;
  float fmSumsqr = 0;
  
  float avgItemsPairwiseSim = 0;

  r_us += gBias; //global bias
  r_us += uBias(user); //user bias
  
  float sumItemBias = 0;
  float sumItemRatings = 0;
  int sz = items.size();
  int nItemPairs = (sz*(sz -1))/2;

  //add item biases
  for (auto&& item: items) {
    sumItemBias += iBias(item);
  }
  sumItemBias = sumItemBias/sz;
  r_us += sumItemBias;
  
  for (int k = 0; k < facDim; k++) {
    float tempSqrsum = 0;
    
    //go over items
    for (auto&& item: items) {
      tempSqrsum += V.row(item)[k];
      fmSumsqr += V.row(item)[k] * V.row(item)[k];
    
      //user
      sumItemRatings += U.row(user)[k]*V.row(item)[k];
    } 

    tempSqrsum = tempSqrsum*tempSqrsum;
    fmSqrSum += tempSqrsum;
  }

  sumItemRatings = sumItemRatings/sz;
  r_us += sumItemRatings;

  avgItemsPairwiseSim = 0.5*(fmSqrSum - fmSumsqr);
  avgItemsPairwiseSim = avgItemsPairwiseSim/nItemPairs;
  r_us += uDivWt(user)*avgItemsPairwiseSim;

  return r_us;
}


float ModelFMUWt::estSetRating(int user, std::vector<int>& items, 
    Eigen::VectorXf& sumItemFactors, float& avgItemsPairwiseSim) {
  
  float r_us = 0;
  float fmSqrSum = 0;
  float fmSumsqr = 0;
  
  sumItemFactors.fill(0);
  avgItemsPairwiseSim = 0;

  r_us += gBias; //global bias
  r_us += uBias(user); //user bias
  
  float sumItemBias = 0;
  float sumItemRatings = 0;
  int sz = items.size();
  int nItemPairs = (sz*(sz -1))/2;

  //add item biases
  for (auto&& item: items) {
    sumItemBias += iBias(item);
  }
  sumItemBias = sumItemBias/sz;
  r_us += sumItemBias;
  
  for (int k = 0; k < facDim; k++) {
    float tempSqrsum = 0;
    
    //go over items
    for (auto&& item: items) {
      tempSqrsum += V.row(item)[k];
      fmSumsqr += V.row(item)[k] * V.row(item)[k];
      sumItemFactors[k] += V.row(item)[k];
    
      //user
      sumItemRatings += U.row(user)[k]*V.row(item)[k];
    } 

    tempSqrsum = tempSqrsum*tempSqrsum;
    fmSqrSum += tempSqrsum;
  }

  sumItemRatings = sumItemRatings/sz;
  r_us += sumItemRatings;

  avgItemsPairwiseSim = 0.5*(fmSqrSum - fmSumsqr);
  avgItemsPairwiseSim = avgItemsPairwiseSim/nItemPairs;
  r_us += uDivWt(user)*avgItemsPairwiseSim;

  return r_us;
}


float ModelFMUWt::estItemRating(int user, int item) {
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


float ModelFMUWt::objective(const std::vector<UserSets>& uSets, 
    gk_csr_t *mat) {
  return Model::objective(uSets, mat) + gBiasReg*gBias*gBias;
}


float ModelFMUWt::objective(const std::vector<UserSets>& uSets) {
  return Model::objective(uSets) + gBiasReg*gBias*gBias;
}


void ModelFMUWt::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelFMUWt::train" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
 
  float avgItemsPairwiseSim;
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
        int sz = items.size();
        int nPairs = (sz*(sz-1)) / 2;

        if (items.size() == 0) {
          std::cerr << "!! zero size itemset found !!" << std::endl; 
          continue;
        }

        //estimate rating on the set & sum of item factors
        float r_us_est = estSetRating(user, items, sumItemFactors, 
            avgItemsPairwiseSim);

        //update items
        for (auto&& item: items) {
          grad = (uDivWt(user)*(sumItemFactors - V.row(item).transpose()))/nPairs;
          grad += U.row(user)/sz;
          grad *= (2.0*(r_us_est - r_us));
          grad += 2.0*iReg*V.row(item).transpose(); 
          V.row(item) -= learnRate*(grad.transpose());
          //update item bias
          iBias(item) -= learnRate*((2.0*(r_us_est - r_us)/sz) 
              + 2.0*iBiasReg*iBias(item));
        }

        //user gradient
        grad = sumItemFactors/sz;
        grad = (2.0*(r_us_est - r_us))*grad
                + 2.0*uReg*U.row(user).transpose();

        //update user
        U.row(user) -= learnRate*(grad.transpose());
        
        //update user bias
        uBias(user) -= learnRate*((2.0*(r_us_est - r_us)) + 2.0*uBiasReg*uBias(user));

       
        //update user div wt
        uDivWt(user) -= learnRate*((2.0*(r_us_est - r_us)/sz))*avgItemsPairwiseSim;

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

