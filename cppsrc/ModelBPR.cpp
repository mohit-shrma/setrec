#include "ModelBPR.h"

float ModelBPR::estItemRating(int user, int item) {
  bool uFound = false, iFound = false;
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


void ModelBPR::train(const Data& data, const Params& params, Model& bestModel) {
  std::cout << "ModelBPR::train" << std::endl;
  
  int u, posItem, negItem;
  float r_ui, r_ui_est, r_uj_est, r_uij_est;
  float expCoeff;

  auto usersNItems  = getUserItems(data.partTrainMat);
  trainUsers = usersNItems.first;
  trainItems = usersNItems.second;
  std::cout << "Train users: " << trainUsers.size() 
    << " items: " << trainItems.size() << std::endl;

  //initialize random engine
  std::mt19937 mt(params.seed);

  auto uiRatings = getUIRatingsTup(data.partTrainMat);

  for (iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    for (auto&& uiRating: uiRatings) {
      //get user item and rating
      u       = std::get<0>(uiRating);
      posItem = std::get<1>(uiRating);
      r_ui    = std::get<2>(uiRating);
      
      //sample neg item or item with lower rating than r_ui
      //TODO: check if csr contains item with zero ratings
      negItem = sampleNegItem(data.trainMat, u, r_ui, mt);

      if (-1 == negItem) {
        continue;
      }
      
      r_ui_est = estItemRating(u, posItem);
      r_uj_est = estItemRating(u, negItem);
      r_uij_est = r_uij_est - r_uj_est;
      expCoeff = -1.0/(1.0 + exp(r_uij_est));

      //update user latent facor
      U.row(u) -= learnRate*( expCoeff(V.row(posItem) - V.row(negItem)) + 2*uReg*U.row(u));
      
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
    if (iter % OBJ_ITER == 0 || iter == param.maxIter -1) {
        
    }

  }


}



