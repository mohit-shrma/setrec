#include "ModelBPRTop.h"

bool ModelBPRTop::isTerminatePrecisionModel(Model& bestModel, const Data& data,
    std::vector<std::vector<std::pair<int, float>>> testRatings,
    int iter, int& bestIter, float& bestValRecall, float& prevValRecall) {

  bool ret = false;
  float currValRecall = corrOrderedItems(testRatings, 3);
  
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

void ModelBPRTop::train(const Data& data, const Params& params, Model& bestModel) {
  std::cout << "ModelBPR::train" << std::endl;
  
  int u, posItem, negItem, iter, bestIter;
  float r_ui, r_ui_est, r_uj_est, r_uij_est;
  float expCoeff, prevValRecall, bestValRecall;

  auto usersNItems  = getUserItems(data.partTrainMat);
  trainUsers = usersNItems.first;
  trainItems = usersNItems.second;
  std::cout << "Train users: " << trainUsers.size() 
    << " items: " << trainItems.size() << std::endl;

  //initialize random engine
  std::mt19937 mt(params.seed);

  auto uiRatings = getUIRatingsTup(data.partTrainMat, 3.0);
  std::cout << "nUIRatings: " << uiRatings.size() << std::endl;

  auto testRatings = getUIRatings(data.partTestMat, data.partValMat, nUsers);
  
  std::unordered_set<int> updUsers; 
  
  //TODO: add convergence criteria which looks for ordering between >3 and <3
  //rated items

  for (iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    int skippedCount = 0;
    for (auto&& uiRating: uiRatings) {
      //get user item and rating
      u       = std::get<0>(uiRating);
      posItem = std::get<1>(uiRating);
      r_ui    = std::get<2>(uiRating);
      
      //sample neg item or item with lower rating than r_ui
      negItem = sampleNegItem(data.partTrainMat, u, r_ui, mt, 3.0);

      if (-1 == negItem) {
        skippedCount++;
        continue;
      }
      
      updUsers.insert(u);

      r_ui_est = estItemRating(u, posItem);
      r_uj_est = estItemRating(u, negItem);
      r_uij_est = r_ui_est - r_uj_est;
      expCoeff = -1.0/(1.0 + exp(r_uij_est));

      //update user latent facor
      U.row(u) -= learnRate*( expCoeff*(V.row(posItem) - V.row(negItem)) + 2*uReg*U.row(u));
      
      //update pos item
      //latent factor
      V.row(posItem) -= learnRate*(expCoeff*U.row(u) + 2.0*iReg*V.row(posItem));
      //bias
      iBias(posItem) -= learnRate*(expCoeff + 2.0*iBiasReg*iBias(posItem));

      //update neg item
      //latent factor
      V.row(negItem) -= learnRate*(-expCoeff*U.row(u) + 2.0*iReg*V.row(negItem));
      iBias(negItem) -= learnRate*(-expCoeff + 2.0*iBiasReg*iBias(negItem));
    }

    //convergence check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter -1) {
      if (isTerminatePrecisionModel(bestModel, data, testRatings, iter, bestIter, 
            bestValRecall, prevValRecall)) {
        break;
      }
      if (0 == iter % 50 || iter == params.maxIter - 1) {
        std::cout << "Iter: " << iter 
          << " val: " << prevValRecall 
          << " bestVal: " << bestValRecall 
          << " bestIter: " << bestIter 
          << " skippedCount: " << skippedCount 
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

