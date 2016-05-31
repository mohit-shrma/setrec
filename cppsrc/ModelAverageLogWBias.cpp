#include "ModelAverageLogWBias.h"


float ModelAverageLogWBias::estItemRating(int user, int item) {
  bool uFound = false, iFound = true;
  float rating = 0;
  if (trainUsers.find(user) != trainUsers.end()) {
    uFound = true;
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


void ModelAverageLogWBias::train(const Data& data, const Params& params,
    Model& bestModel) {
  
  std::cout << "ModelAverageLogWBias::train" << std::endl;

  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  float bestRecall, prevRecall, bestValRecall, prevValRecall;
  int bestIter;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
  int nTrUsers = (int)uInds.size(); 

  //initialize random engine
  std::mt19937 mt(params.seed);

  trainUsers = data.trainUsers; 
  trainItems = data.trainItems;
  std::cout << "train Users: " << trainUsers.size() 
    << " trainItems: " << trainItems.size() << std::endl;
  
  for (int iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uInds.begin(), uInds.end(), mt);
    int skippedCount = 0;
    for (int i = 0; i < data.nTrainSets/nTrUsers; i++) {

      for (auto&& uInd: uInds) {
        UserSets uSet = data.trainSets[uInd];
        int user = uSet.user;
        
        if (invalidUsers.find(user) != invalidUsers.end()) {
          continue;
        }        

        if (uSet.itemSets.size() == 0) {
          std::cerr << "!! zero size user itemset found !! " << user << std::endl; 
          continue;
        }
        
        //sample high and low set ind for the user
        auto hiLo = uSet.sampPosNeg(mt);
       
        int hiSetInd = hiLo.first;
        int loSetInd = hiLo.second;

        if (-1 == hiSetInd || -1 == loSetInd) {
          //cant sample high and low sets for user
          //std::cerr << "cant sample high and low sets for user" << std::endl;
          skippedCount++;
          invalidUsers.insert(user);
          continue;
        }

        //get high items and set rating
        auto hiItems   = uSet.itemSets[hiSetInd].first;
        float r_us     = uSet.itemSets[hiSetInd].second;
        float r_us_est = estSetRating(user, hiItems);

        //get low items and set rating
        auto loItems   = uSet.itemSets[loSetInd].first;
        float r_ut     = uSet.itemSets[loSetInd].second;
        float r_ut_est = estSetRating(user, loItems);

        float r_ust_est_diff = r_us_est - r_ut_est; 
        float r_ust_diff = r_us - r_ut;
        
        //if (gamma + r_ust_diff - r_ust_est_diff > 0) {
        float coeff = gamma + r_ust_diff - r_ust_est_diff;  
        float expCoeff = 1.0/(1.0 + exp(-1.0*coeff));

        //compute sum of item latent factors
        sumItemFactors.fill(0);    
        for (auto item: hiItems) {
          sumItemFactors += V.row(item);
        }
        for (auto item: loItems) {
          sumItemFactors -= V.row(item);
        }  
        
        //user gradient
        grad = -expCoeff*sumItemFactors/hiItems.size() 
          + 2.0*uReg*U.row(user).transpose();
      
        //update user
        U.row(user) -= learnRate*(grad.transpose());
      
        //update items
        grad = U.row(user)/hiItems.size();
      
        std::unordered_set<int> hiItemsSet(hiItems.begin(), hiItems.end());
        std::unordered_set<int> loItemsSet(loItems.begin(), loItems.end());
        for (auto&& item: hiItemsSet) {
          //check if item occurs in lo itemset
          if (loItemsSet.find(item) != loItemsSet.end()) {
            //found
            continue;
          }
          tempGrad = -expCoeff*grad + 2.0*iReg*V.row(item).transpose(); 
          V.row(item) -= learnRate*(tempGrad.transpose());
          iBias(item) -= learnRate*(-expCoeff/hiItems.size() 
              + 2.0*iBiasReg*iBias(item));
        }
        for (auto&& item: loItemsSet) {
          //check if item occurs in lo itemset
          if (hiItemsSet.find(item) != hiItemsSet.end()) {
            //found
            continue;
          }
          tempGrad = grad + 2.0*iReg*V.row(item).transpose(); 
          V.row(item) -= learnRate*(tempGrad.transpose());
          iBias(item) -= learnRate*(expCoeff/loItems.size() 
              + 2.0*iBiasReg*iBias(item));
        }
      

      }
    }    
    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateRankSetModel(bestModel, data, iter, bestIter, 
            prevValRecall, bestValRecall)) {
        break;
      }
      if (iter % 10 == 0 || iter == params.maxIter-1) {
        std::cout << "Skipped: " << skippedCount <<  " invalid users: " 
          << invalidUsers.size() << std::endl;
        std::cout << "Iter:" << iter 
          << " val: " << prevValRecall 
          << " bestVal: " << bestValRecall
          << " bestIter: " <<bestIter
          << std::endl;
      }
    }

  }

}





