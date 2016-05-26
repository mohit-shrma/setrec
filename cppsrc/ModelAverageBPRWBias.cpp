#include "ModelAverageBPRWBias.h"

float ModelAverageBPRWBias::estItemRating(int user, int item) {
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

void ModelAverageBPRWBias::train(const Data& data, const Params& params,
    Model& bestModel) {
  
  std::cout << "ModelAverageBPRWBias::train" << std::endl;
  
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

  auto usersNItems = getUserItems(data.trainSets);
  trainUsers = usersNItems.first;
  trainItems = usersNItems.second;
  std::cout << "train Users: " << trainUsers.size() 
    << " trainItems: " << trainItems.size() << std::endl;

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
        
        if (invalidUsers.find(user) != invalidUsers.end()) {
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
      if (isTerminateRecallModel(bestModel, data, iter, bestIter, bestRecall, prevRecall,
            bestValRecall, prevValRecall)) {
        break;
      }
      if (iter % 10 == 0 || iter == params.maxIter - 1) {
        std::cout << "Skipped: " << skippedCount <<  " invalid users: " 
          << invalidUsers.size() << std::endl;
        std::cout << "Iter:" << iter 
          << " REC@10 train: " << prevRecall 
          << " val: " << prevValRecall << " bestVal: " << bestValRecall
          << " test: " << ratingsNDCG(data.testURatings)
          << std::endl;
      }
    }

  }

}

