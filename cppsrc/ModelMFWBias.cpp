#include "ModelMFWBias.h"

float ModelMFWBias::isTerminateModelIRMSE(Model& bestModel, const Data& data,
    int iter, int& bestIter, float& bestObj, float& prevObj, float& bestValRMSE,
    float& prevValRMSE) {
  
  bool ret = false;  
  float currObj = objective(data.partTrainMat);
  float currValRMSE = -1;
  
  currValRMSE = rmse(data.partValMat); 

  if (iter > 0) {
    if (currValRMSE < bestValRMSE) {
      bestModel = *this;
      bestValRMSE = currValRMSE;
      bestIter = iter;
      bestObj = currObj;
    } 
  
    if (iter - bestIter >= CHANCE_ITER) {
      //cant improve validation RMSE
      std::cout << "NOT CONVERGED VAL: bestIter:" << bestIter << " bestObj:" 
        << bestObj << " bestValRMSE: " << bestValRMSE << " currIter:"
        << iter << " currObj: " << currObj << " currValRMSE:" 
        << currValRMSE << std::endl;
      ret = true;
    }
    
    if (fabs(prevObj - currObj) < EPS) {
      //objective converged
      std::cout << "CONVERGED OBJ:" << iter << " currObj:" << currObj 
        << " bestValRMSE:" << bestValRMSE;
      ret = true;
    }
  }
  
  if (0 == iter) {
    bestObj = currObj;
    bestValRMSE = currValRMSE;
    bestIter = iter;
  }
  
  prevObj     = currObj;
  prevValRMSE = currValRMSE;

  return ret;
}


float ModelMFWBias::estItemRating(int user, int item) {
  bool uFound = false, iFound = false;
  float rating = gBias;
  if (trainUsers.find(user) != trainUsers.end()) {
    uFound = true;
    //rating += uBias(user);
  }
  if (trainItems.find(item) != trainItems.end()) {
    iFound = true;
    //rating += iBias(item);
  }
  if (uFound && iFound) {
    rating += U.row(user).dot(V.row(item));
  }
  return rating;
}


void ModelMFWBias::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelMFWBias::train" << std::endl;
  
  int u, item, bestIter;
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  float r_ui, r_ui_est, diff;

  //initialize user-item factors with SVD
  svdFrmSvdlibCSR(data.partTrainMat, facDim, U, V, false);
 
  //initialize global bias
  gBias = meanRating(data.partTrainMat);

  std::cout << "Train RMSE: " << rmse(data.partTrainMat) << std::endl;

  auto uiRatings = getUIRatingsTup(data.partTrainMat);
  auto usersNItems  = getUserItems(data.partTrainMat);
  trainUsers = usersNItems.first;
  trainItems = usersNItems.second;
  std::cout << "Train users: " << trainUsers.size() 
    << " items: " << trainItems.size() << std::endl;
  //initialize random engine
  std::mt19937 mt(params.seed);

  for (int iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
    for (auto&& uiRating : uiRatings) {
      //get user, item and ratings
      u       = std::get<0>(uiRating);
      item    = std::get<1>(uiRating);
      r_ui    = std::get<2>(uiRating);
      r_ui_est = estItemRating(u, item);

      diff = r_ui_est - r_ui;
      
      //update user bias
      //uBias(u) -= learnRate*(2.0*diff + 2.0*uBiasReg*uBias(u)); 
      //update user latent factor
      U.row(u) -= learnRate*(2.0*diff*V.row(item) + 2.0*uReg*U.row(u));
      //update item bias
      //iBias(item) -= learnRate*(2.0*diff + 2.0*iBiasReg*iBias(item));
      //update item latent factor
      V.row(item) -= learnRate*(2.0*diff*U.row(u) + 2.0*iReg*V.row(item));
    }
    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter - 1) {
      if (isTerminateModelIRMSE(bestModel, data, iter, bestIter, bestObj, 
          prevObj, bestValRMSE, prevValRMSE)) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      if (iter % 100 == 0) {
        std::cout << "Iter: " << iter << " obj: " << prevObj << " valRMSE: "
          << prevValRMSE << " best valRMSE: " << bestValRMSE 
          << " trainRMSE: " << rmse(data.partTrainMat) << " testRMSE: " 
          << rmse(data.partTestMat) << std::endl;
        //bestModel.save(params.prefix);
      }
    }

  }

}



