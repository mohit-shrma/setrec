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
  
  int u, item;
  float r_ui;

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
      item    = std::get<1>(uiRating);
      r_ui    = std::get<2>(uiRating);
      
      //sample neg item or item with lower rating than r_ui

    }

  }


}



