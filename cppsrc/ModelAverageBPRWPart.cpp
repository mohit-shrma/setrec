#include "ModelAverageBPRWPart.h"

bool ModelAverageBPRWPart::isTerminatePrecisionModel(Model& bestModel, const Data& data,
    int iter, int& bestIter, float& bestValRecall, float& prevValRecall) {

  bool ret = false;
  float currValRecall = corrOrderedItems(data.partValMat, TOP_RAT_THRESH);
  
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


void ModelAverageBPRWPart::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelAverageBPRWPart::train" << std::endl;
    
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  float bestRecall, prevRecall, bestValRecall, prevValRecall;
  float r_ui_est, r_uj_est, r_uij_est;
  int bestIter;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
  int nTrUsers = (int)uInds.size(); 

  //initialize random engine
  std::mt19937 mt(params.seed);
  std::uniform_int_distribution<int> dist(0, 1000);

  auto usersNItems = getUserItems(data.trainSets);
  trainUsers = usersNItems.first;
  trainItems = usersNItems.second;
  std::cout << "train Users: " << trainUsers.size() 
    << " trainItems: " << trainItems.size() << std::endl;
  
  //get ratings > TOP_RAT_THRESH
  auto uiRatings = getUIRatingsTup(data.partTrainMat, TOP_RAT_THRESH);
  std::cout << "nUIRatings: " << uiRatings.size() << std::endl;

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

        //sample pos/neg item for user
        hiLo = samplePosNegItem(data.partTrainMat, user, mt, TOP_RAT_THRESH);
        int posItem = hiLo.first;
        int negItem = hiLo.second;
        if (-1 == posItem || -1 == negItem) {
          continue;
        }
        r_ui_est = estItemRating(user, posItem);
        r_uj_est = estItemRating(user, negItem);
        r_uij_est = r_ui_est - r_uj_est;
        float expCoeff = -1.0/(1.0 + exp(r_uij_est));

        //update user latent facor
        U.row(user) -= learnRate*( expCoeff*(V.row(posItem) - V.row(negItem)) + 2*uReg*U.row(user));
        
        //update pos item
        //latent factor
        V.row(posItem) -= learnRate*(expCoeff*U.row(user) + 2.0*iReg*V.row(posItem));
        //bias
        iBias(posItem) -= learnRate*(expCoeff + 2.0*iBiasReg*iBias(posItem));

        //update neg item
        //latent factor
        V.row(negItem) -= learnRate*(-expCoeff*U.row(user) + 2.0*iReg*V.row(negItem));
        iBias(negItem) -= learnRate*(-expCoeff + 2.0*iBiasReg*iBias(negItem));

      }
    }    
    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      //TODO: how to terminate as item ranking should also be taken into account
      //if (isTerminateRankSetModel(bestModel, data, iter, bestIter, 
      //      prevValRecall, bestValRecall, TOP_RAT_THRESH)) {
      //  break;
      //}
      if (isTerminatePrecisionModel(bestModel, data, iter, bestIter, 
            bestValRecall, prevValRecall)) {
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


