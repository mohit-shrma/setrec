#include "ModelAverageSigmoidWBias.h"


float ModelAverageSigmoidWBias::estSetRating(int user, std::vector<int>& items) {
  float r_us_est = 0;
  
  for (auto&& item: items) {
    r_us_est += estItemRating(user, item);
  }
  r_us_est = r_us_est/items.size();

  float dev = r_us_est - uDivWt(user);
  r_us_est = sigmoid(dev, g_k);

  return r_us_est;
}


float ModelAverageSigmoidWBias::objective(const std::vector<UserSets>& uSets, 
    gk_csr_t *mat) {
  float obj = ModelAverageWBias::objective(uSets, mat);
  for (int user = 0; user < nUsers; user++) {
    obj += uDivWt(user)*uDivWt(user)*uSetBiasReg;
  } 
  return obj;
}


float ModelAverageSigmoidWBias::objective(const std::vector<UserSets>& uSets) {
  float obj = ModelAverageWBias::objective(uSets);
  for (int user = 0; user < nUsers; user++) {
    obj += uDivWt(user)*uDivWt(user)*uSetBiasReg;
  } 
  return obj;
}


void ModelAverageSigmoidWBias::train(const Data& data, const Params& params, 
    Model& bestModel) {
  
  std::cout << "ModelAverageSigmoidWBias::train" << std::endl;

  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  std::cout << "Train RMSE: " << rmse(data.trainSets) << std::endl;
 
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

  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  std::vector<int> items; 
  //initialize random engine
  std::mt19937 mt(params.seed);
  std::uniform_int_distribution<int> dist(0, 1000);

  for (int iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uInds.begin(), uInds.end(), mt);
    for (int i = 0; i < data.nTrainSets/nTrUsers; i++) {
      for (auto&& uInd: uInds) {
        UserSets uSet = data.trainSets[uInd];
        int user = uSet.user;
              
        //select a set at random
        int setInd = dist(mt) % uSet.itemSets.size();
        items = uSet.itemSets[setInd].first;
        float r_us = uSet.itemSets[setInd].second;

        //estimate rating on the set and latent factor sum
        float r_us_est = 0;
        sumItemFactors.fill(0);
        for (auto&& item: items) {
          r_us_est += iBias(item) + estItemRating(user, item);
          sumItemFactors += V.row(item);
        }
        r_us_est = r_us_est/items.size();
        r_us_est += uBias(user);

        float dev = r_us_est - uDivWt(user);
        r_us_est = sigmoid(dev, g_k);

        float commGradCoeff = exp(-1.0*g_k*dev)*r_us_est*r_us_est;
        commGradCoeff = 2*(r_us_est - r_us)*commGradCoeff;

        //user gradient
        grad = ((g_k*commGradCoeff)/items.size())*sumItemFactors;
        
        //update user
        U.row(user) -= learnRate*(grad.transpose() + 2.0*uReg*U.row(user));
        uBias(user) -= learnRate*(g_k*commGradCoeff + 2.0*uBiasReg*uBias(user));

        //update items
        grad = ((g_k*commGradCoeff)/items.size())*U.row(user);
        for (auto&& item: items) {
          V.row(item) -= learnRate*(grad.transpose() + 2.0*iReg*V.row(item));
          iBias(item) -= learnRate*((g_k*commGradCoeff)/items.size() 
              + 2.0*iBiasReg*iBias(item));
        }

        //u_m gradient
        float u_mGrad = commGradCoeff*-1.0*g_k + 2.0*uDivWt(user)*uSetBiasReg;
        //update u_m
        uDivWt(user) -= learnRate*u_mGrad;
       
        /*
        //compute g_k grad
        float g_kGrad = commGradCoeff*dev + 2.0*g_k*g_kReg;
        //update g_k
        g_k -= learnRate*g_kGrad;
        */
      }
    } 
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE)) {
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

}
