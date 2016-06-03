#include "ModelAverageSetBiasWPart.h"

float ModelAverageSetBiasWPart::estSetRating(int user, 
    std::vector<int>& items) {
  float ratSum = 0;

  for (auto&& item: items) {
    ratSum += estItemRating(user, item);
  }

  ratSum = ratSum/items.size();

  ratSum += uSetBias(user);

  return ratSum;
}


float ModelAverageSetBiasWPart::objective(const std::vector<UserSets>& uSets,
    gk_csr_t *mat) {
  
  float norm;
  float obj = ModelAverageWPart::objective(uSets, mat);

  //add set bias reg
  norm = uSetBias.norm();
  obj += uSetBiasReg*norm*norm;

  return obj;
}


void ModelAverageSetBiasWPart::train(const Data& data, const Params& params, 
    Model& bestModel) {

  std::cout << "ModelAverageSetBiasWPart::train" << std::endl;
  
  //initialize user-item factors with SVD
  svdFrmSvdlibCSR(data.partTrainMat, facDim, U, V, false);
  
  std::cout << "Train RMSE: " << rmse(data.trainSets) << std::endl;
  
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  float bestObj, prevObj, bestValRMSE, prevValRMSE;
  float uBiasGrad, iBiasGrad;
  int bestIter;

  std::vector<int> uInds(data.trainSets.size());
  std::iota(uInds.begin(), uInds.end(), 0);
  int nTrUsers = (int)uInds.size(); 

  auto partUIRatings = getUIRatings(data.partTrainMat);

  trainUsers = data.trainUsers;
  trainItems = data.trainItems;
  std::cout << "train Users: " << trainUsers.size() 
    << " trainItems: " << trainItems.size() << std::endl;

  //initialize random engine
  std::mt19937 mt(params.seed);
  std::uniform_int_distribution<int> dist(0, 1000);

  for (int iter = 0; iter < params.maxIter; iter++) {
    std::shuffle(uInds.begin(), uInds.end(), mt);
    for (int i = 0; i < data.nTrainSets/nTrUsers; i++) {
      for (auto&& uInd: uInds) {
        const UserSets& uSet = data.trainSets[uInd];
        int user = uSet.user;
              
        if (uSet.itemSets.size() == 0) {
          std::cerr << "!! zero size user itemset found !! " << user << std::endl; 
          continue;
        }
        //select a set at random
        int setInd = dist(mt) % uSet.itemSets.size();
        auto items = uSet.itemSets[setInd].first;
        float r_us = uSet.itemSets[setInd].second;

        if (items.size() == 0) {
          std::cerr << "!! zero size itemset foundi !!" << std::endl; 
          continue;
        }

        //estimate rating on the set
        float r_us_est = estSetRating(user, items);

        //compute sum of item latent factors and tempgrad if rating is in partmat
        sumItemFactors.fill(0);    
        tempGrad.fill(0);
        uBiasGrad = 0;
        for (auto item: items) {
          sumItemFactors += V.row(item);
          if (partUIRatings[user].find(item) != partUIRatings[user].end()) {
             //found in partial rating matrix
             float r_ui_est = estItemRating(user, item);
             float r_ui = partUIRatings[user][item];
             tempGrad += 2.0*(r_ui_est - r_ui)*V.row(item);
             uBiasGrad += 2.0*(r_ui_est - r_ui);
          }
        }

        //user gradient
        grad = (2.0*(r_us_est - r_us)/items.size())*sumItemFactors
                + 2.0*uReg*U.row(user).transpose();
        grad += tempGrad;

        //update user
        U.row(user) -= learnRate*(grad.transpose());
        
        //update user bias
        uBias(user) -= learnRate*((2.0*(r_us_est - r_us) + uBiasGrad) + 2.0*uBiasReg*uBias(user));
        
        //update user set bias
        uSetBias(user) -= learnRate*(2.0*(r_us_est - r_us) 
            + 2.0*uSetBiasReg*uSetBias(user));
        
        //update items
        grad = (2.0*(r_us_est - r_us)/items.size())*U.row(user);
        for (auto&& item: items) {
          tempGrad = grad + 2.0*iReg*V.row(item).transpose();
          iBiasGrad = 2.0*(r_us_est - r_us)/items.size();
          if (partUIRatings[user].find(item) != partUIRatings[user].end()) {
            //found in partial rating matrix
            float r_ui_est = estItemRating(user, item);
            float r_ui = partUIRatings[user][item];
            tempGrad += 2.0*(r_ui_est - r_ui)*U.row(user);
            iBiasGrad += 2.0*(r_ui_est - r_ui);
          } 

          V.row(item) -= learnRate*(tempGrad.transpose());
          
          //update item bias
          iBias(item) -= learnRate*(iBiasGrad + 2.0*iBiasReg*iBias(item));
        }

      }
    }    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModelWPartIRMSE(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE)) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      if (iter % 10 == 0 || iter == params.maxIter -1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << std::endl;
        //bestModel.save(params.prefix);
      }
    }
  }

}




