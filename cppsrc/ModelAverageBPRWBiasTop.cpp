#include "ModelAverageBPRWBiasTop.h"


bool ModelAverageBPRWBiasTop::isTerminateRankSetModel(Model& bestModel, 
    const Data& data, int iter, int& bestIter, float& prevValRecall,
    float& bestValRecall, float lb) {
  bool ret = false;
  float currValRecall = fracCorrOrderedSets(data.testValMergeSets, lb);
  
  if (iter > 0) {
    if (currValRecall > bestValRecall) {
      bestModel     = *this;
      bestValRecall = currValRecall;
      bestIter      = iter;
    } 
    
    if (iter - bestIter >= CHANCE_ITER) {
      //cant improve validation RMSE
      std::cout << "NOT CONVERGED VAL: bestIter:" << bestIter  
        << " bestValRecall: " << bestValRecall << " currIter: "
        << iter << " currValRecall: " << currValRecall << std::endl;
      ret = true;
    }
  
  }

  if (0 == iter) {
    bestValRecall = currValRecall;
    bestIter      = iter;
    bestModel     = *this;
  }

  prevValRecall = currValRecall;
  return ret;


}


void ModelAverageBPRWBiasTop::train(const Data& data, const Params& params, 
    Model& bestModel) {
    
  std::cout << "ModelAverageBPRWBiasTop::train" << std::endl;
  
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  float bestValRecall, prevValRecall;
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

  std::unordered_set<int> updUsers; 
  
  for (int iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uInds.begin(), uInds.end(), mt);
    int skippedCount = 0;
    for (int i = 0; i < data.nTrainSets/nTrUsers; i++) {

      for (auto&& uInd: uInds) {
        UserSets uSet = data.trainSets[uInd];
        int user = uSet.user;
        
        if (uSet.itemSets.size() == 0) {
          std::cerr << "!! zero size user itemset found !! " << user << std::endl; 
          continue;
        }
        
        //sample high and low set ind for the user
        //r_us <= 3, r_ut > 3
        auto hiLo = uSet.sampPosNeg(mt, 3);
       
        int hiSetInd = hiLo.first;
        int loSetInd = hiLo.second;

        if (-1 == hiSetInd || -1 == loSetInd) {
          //cant sample high and low sets for user
          //std::cerr << "cant sample high and low sets for user" << std::endl;
          skippedCount++;
          continue;
        }
        
        updUsers.insert(user);

        //get high items and set rating
        auto hiItems = uSet.itemSets[hiSetInd].first;
        float r_us_est = estSetRating(user, hiItems);

        //get low items and set rating
        auto loItems = uSet.itemSets[loSetInd].first;
        float r_ut_est = estSetRating(user, loItems);
        
        //TODO: experiment by commenting or uncommenting
        //if (r_us_est > r_us_est) {
        //  continue;
        //}

        float r_ust_diff = r_us_est - r_ut_est; 
        float expDiff = -1.0/(1.0 + exp(r_ust_diff));

        //compute sum of item latent factors
        sumItemFactors.fill(0);    
        for (auto item: hiItems) {
          sumItemFactors += V.row(item);
        }
        for (auto item: loItems) {
          sumItemFactors -= V.row(item);
        }

        //user gradient
        grad = (expDiff/hiItems.size())*sumItemFactors
                + 2.0*uReg*U.row(user).transpose();

        //update user
        U.row(user) -= learnRate*(grad.transpose());
        
        //update items
        grad = (expDiff/hiItems.size())*U.row(user);
        
        std::unordered_set<int> hiItemsSet(hiItems.begin(), hiItems.end());
        std::unordered_set<int> loItemsSet(loItems.begin(), loItems.end());
        for (auto&& item: hiItemsSet) {
          //check if item occurs in lo itemset
          if (loItemsSet.find(item) != loItemsSet.end()) {
            //found
            continue;
          }
          tempGrad = grad + 2.0*iReg*V.row(item).transpose(); 
          V.row(item) -= learnRate*(tempGrad.transpose());
          iBias(item) -= learnRate*(expDiff/hiItems.size() 
              + 2.0*iBiasReg*iBias(item));
        
        }
        for (auto&& item: loItemsSet) {
          //check if item occurs in lo itemset
          if (hiItemsSet.find(item) != hiItemsSet.end()) {
            //found
            continue;
          }
          tempGrad = -grad + 2.0*iReg*V.row(item).transpose(); 
          V.row(item) -= learnRate*(tempGrad.transpose());
          iBias(item) -= learnRate*(-expDiff/loItems.size() 
              + 2.0*iBiasReg*iBias(item));
        }

      }
    }    
    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateRankSetModel(bestModel, data, iter, bestIter, 
            prevValRecall, bestValRecall, 3.0)) {
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
  
  //add non-update users as invalid
  for (auto&& u: trainUsers) {
    if (updUsers.find(u) == updUsers.end()) {
      invalidUsers.insert(u);
    }
  }
  
  std::cout << "No.  of invalid users: " << invalidUsers.size() << std::endl;
 
}


