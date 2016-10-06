#include "ModelFMUWtBPR.h"


float ModelFMUWtBPR::estSetRating(int user, std::vector<int>& items) {
  
  float r_us = 0;
  float fmSqrSum = 0;
  float fmSumsqr = 0;
  
  float avgItemsPairwiseSim = 0;

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


float ModelFMUWtBPR::estSetRating(int user, std::vector<int>& items, 
    Eigen::VectorXf& sumItemFactors, float& avgItemsPairwiseSim) {
  
  float r_us = 0;
  float fmSqrSum = 0;
  float fmSumsqr = 0;
  
  sumItemFactors.fill(0);
  avgItemsPairwiseSim = 0;

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


void ModelFMUWtBPR::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelFMUWtBPR::train" << std::endl; 
 
  Eigen::VectorXf s_sumItemFactors(facDim);
  float s_avgItemsPairwiseSim;
  Eigen::VectorXf t_sumItemFactors(facDim);
  float t_avgItemsPairwiseSim;
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
        //r_us <= TOP_RAT_THRESH, r_ut > TOP_RAT_THRESH
        auto hiLo = uSet.sampPosNeg(mt, TOP_RAT_THRESH);
        
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
        float r_us_est = estSetRating(user, hiItems, s_sumItemFactors,
            s_avgItemsPairwiseSim);

        //get low items and set rating
        auto loItems = uSet.itemSets[loSetInd].first;
        float r_ut_est = estSetRating(user, loItems, t_sumItemFactors,
            t_avgItemsPairwiseSim);

        float r_ust_diff = r_us_est - r_ut_est; 
        float expDiff = -1.0/(1.0 + exp(r_ust_diff));
        int sz = hiItems.size();
        int nPairs = (sz*(sz-1)) / 2; 
     
        //update items
        grad = U.row(user)/sz;

        //update items in s
        for (auto&& item: hiItems) {
          tempGrad = grad + (uDivWt(user)/nPairs)*(s_sumItemFactors 
              - V.row(item).transpose());
          tempGrad = expDiff*tempGrad;
          tempGrad += 2.0*iReg*V.row(item).transpose();
          V.row(item) -= learnRate*tempGrad.transpose();
          iBias(item) -= learnRate*(expDiff/sz + 2.0*iBiasReg*iBias(item));
        }

        //update items in t
        for (auto&& item: loItems) {
          tempGrad = grad + (uDivWt(user)/nPairs)*(t_sumItemFactors 
              - V.row(item).transpose());
          tempGrad = -expDiff*tempGrad;
          tempGrad += 2.0*iReg*V.row(item).transpose();
          V.row(item) -= learnRate*tempGrad.transpose();
          iBias(item) -= learnRate*(-expDiff/sz + 2.0*iBiasReg*iBias(item));
        }

        //update user
        grad = (expDiff/sz) * (s_sumItemFactors - t_sumItemFactors);
        grad += 2.0*uReg*U.row(user).transpose();  
        U.row(user) -= learnRate*grad.transpose();
        
        //update user div wt
        uDivWt(user) -= learnRate*(expDiff*(s_avgItemsPairwiseSim 
              - t_avgItemsPairwiseSim) + 2.0*uSetBiasReg*uDivWt(user));

      }
    }    
    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateRankSetModel(bestModel, data, iter, bestIter, 
            prevValRecall, bestValRecall, TOP_RAT_THRESH)) {
        break;
      }
      
      if (iter%10 == 0 || iter == params.maxIter - 1) {
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
