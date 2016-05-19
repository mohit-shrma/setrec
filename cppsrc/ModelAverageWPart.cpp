#include "ModelAverageWPart.h"

bool ModelAverageWPart::isTerminateModelWPartIRMSE(Model& bestModel, 
    const Data& data, int iter, int& bestIter, float& bestObj, float& prevObj, 
    float& bestValRMSE, float& prevValRMSE) {

  bool ret = false;  
  float currObj = objective(data.trainSets, data.partTrainMat);
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
    
     
    /*
    if (fabs(prevValRMSE - currValRMSE) < EPS) {
      //Validation rmse converged
      std::cout << "CONVERGED VAL: bestIter:" << bestIter << " bestObj:" 
        << bestObj << " bestValRMSE: " << bestValRMSE << " currIter:"
        << iter << " currObj: " << currObj << " currValRMSE:" 
        << currValRMSE << std::endl;
      ret = true;
    }
    */
  }
  
  if (0 == iter) {
    bestObj = currObj;
    bestValRMSE = currValRMSE;
    bestIter = iter;
  }
  
  prevObj = currObj;
  prevValRMSE = currValRMSE;

  return ret;
}


float ModelAverageWPart::objective(const std::vector<UserSets>& uSets,
    gk_csr_t *mat) {
  
  float norm;
  float obj = ModelAverageWBias::objective(uSets, mat);

  //add uBias
  norm = uBias.norm();
  obj += uBiasReg*norm*norm;

  //add iBias
  norm = iBias.norm();
  obj += iBiasReg*norm*norm;
  
  return obj;
}


void ModelAverageWPart::trainJoint(const Data& data, const Params& params,
    Model& bestModel) {

  std::cout << "ModelAverageWPart::trainJoint" << std::endl;
  
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
        bestModel.save(params.prefix);
        break;
      }
      if (iter % 5 == 0 || iter == params.maxIter -1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << " recall@10: " << recallTopN(data.ratMat, data.trainSets, 10)
          << " spearman@10: " << spearmanRankN(data.ratMat, data.trainSets, 10)
          << " invCount@10: " << inversionCount(data.ratMat, data.trainSets, 10)
          << std::endl;
        bestModel.save(params.prefix);
      }
    }


  }




}


void ModelAverageWPart::trainIter(const Data& data, const Params& params,
    Model& bestModel) {

  std::cout << "ModelAverageWPart::trainIter" << std::endl;
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
  auto uiRatings = getUIRatingsTup(data.partTrainMat);

  //initialize random engine
  std::mt19937 mt(params.seed);
  std::uniform_int_distribution<int> dist(0, 1000);

  for (int iter = 0; iter < params.maxIter; iter++) {
    float prevRMSE = 0, currRMSE = 0;
    currRMSE = rmse(data.partTrainMat);
    while (fabs(currRMSE - prevRMSE) > 0.01 ) {
      std::shuffle(uiRatings.begin(), uiRatings.end(), mt);
      for (auto&& uiRating: uiRatings) {
        int user = std::get<0>(uiRating);
        int item = std::get<1>(uiRating);
        float r_ui = std::get<2>(uiRating);
        float r_ui_est = estItemRating(user, item);

        //compute user grad
        tempGrad = 2.0*(r_ui_est - r_ui)*V.row(item) + 2.0*uReg*U.row(user);
        //update user factor
        U.row(user) -= learnRate*tempGrad;
        
        //update user bias
        uBias(user) -= learnRate*(2.0*(r_ui_est - r_ui) + 2.0*uReg*uBias(user));
        
        //compute item grad
        tempGrad = 2.0*(r_ui_est - r_ui)*U.row(user) + 2.0*iReg*V.row(item);
        //update item factor
        V.row(item) -= learnRate*tempGrad;

        //update item-bias
        iBias(item) -= learnRate*(2.0*(r_ui_est - r_ui) + 2.0*iReg*iBias(item));
      }
      prevRMSE = currRMSE;
      currRMSE = rmse(data.partTrainMat);
    }

    currRMSE = rmse(data.trainSets);
    prevRMSE = 0;
    while(fabs(currRMSE - prevRMSE) > 0.01) {
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
          }

          //user gradient
          grad = (2.0*(r_us_est - r_us)/items.size())*sumItemFactors
                  + 2.0*uReg*U.row(user).transpose();
          grad += tempGrad;

          //update user
          U.row(user) -= learnRate*(grad.transpose());
          
          //update user bias
          uBias(user) -= learnRate*((2.0*(r_us_est - r_us) + uBiasGrad) + 2.0*uReg*uBias(user));

          //update items
          grad = (2.0*(r_us_est - r_us)/items.size())*U.row(user);
          for (auto&& item: items) {
            tempGrad = grad + 2.0*iReg*V.row(item).transpose();
            iBiasGrad = 2.0*(r_us_est - r_us)/items.size();
            V.row(item) -= learnRate*(tempGrad.transpose());
            //update item bias
            iBias(item) -= learnRate*(iBiasGrad + 2.0*iReg*iBias(item));
          }

        }
      }    
      prevRMSE = currRMSE;
      currRMSE = rmse(data.trainSets);
    }
    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModelWPart(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE)) {
        //save best model
        bestModel.save(params.prefix);
        break;
      }
      std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
        << prevValRMSE << " best val RMSE:" << bestValRMSE 
        << " train RMSE:" << rmse(data.trainSets) 
        << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
        << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
        << " recall@10: " << recallTopN(data.ratMat, data.trainSets, 10)
        << " spearman@10: " << spearmanRankN(data.ratMat, data.trainSets, 10)
        << std::endl;
      bestModel.save(params.prefix);
    }

  }




}


void ModelAverageWPart::train(const Data& data, const Params& params,
    Model& bestModel) {

  trainJoint(data, params, bestModel); 

}
