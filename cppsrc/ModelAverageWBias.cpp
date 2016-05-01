#include "ModelAverageWBias.h"

float ModelAverageWBias::estItemRating(int user, int item) {
  float rating = uBias(user) + iBias(item) + U.row(user).dot(V.row(item));
  return rating;
}


float ModelAverageWBias::estSetRating(int user, std::vector<int>& items) {
  int setSz = items.size();
  float r_us = 0;
  for (auto&& item: items) {
    r_us += estItemRating(user, item);
  }
  r_us = r_us/setSz;// + gBias;
  return r_us;
}


float ModelAverageWBias::objective(const std::vector<UserSets>& uSets) {
  float obj = Model::objective(uSets);
  float norm = 0;
  
  //add biases regularization
  norm = uBias.norm();
  obj += norm*norm*uReg;

  norm = iBias.norm();
  obj += norm*norm*iReg;

  return obj;
}


void ModelAverageWBias::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelAverageWBias::train" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  std::cout << "Train RMSE: " << rmse(data.trainSets) << std::endl;
  
  Eigen::VectorXf sumItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  Eigen::MatrixXf uGradsAcc(nUsers, facDim);
  uGradsAcc.fill(0);
  Eigen::MatrixXf iGradsAcc(nItems, facDim);
  iGradsAcc.fill(0);
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
  gBias = meanSetRating;


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
          std::cerr << "!! zero size user itemset foundi !! " << user << std::endl; 
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

        //compute sum of item latent factors
        sumItemFactors.fill(0);    
        for (auto item: items) {
          sumItemFactors += V.row(item);
        }

        //user gradient
        grad = (2.0*(r_us_est - r_us)/items.size())*sumItemFactors
                + 2.0*uReg*U.row(user).transpose();
        //uGradsAcc.row(user) = uGradsAcc.row(user)*params.rhoRMS 
        //  + (1.0 - params.rhoRMS)*(grad.cwiseProduct(grad).transpose());

        //update user
        U.row(user) -= learnRate*(grad.transpose());
        
        //update rms prop
        //for (int k = 0; k < facDim; k++) {
        //  U(user, k) -= (learnRate/(sqrt(uGradsAcc(user, k) + 0.0000001)))*grad[k];
        //}
        
        //update user bias
        uBias(user) -= learnRate*((2.0*(r_us_est - r_us)/items.size()) 
            + 2.0*uReg*uBias(user));

        //update items
        grad = (2.0*(r_us_est - r_us)/items.size())*U.row(user);
        for (auto&& item: items) {
          tempGrad = grad + 2.0*iReg*V.row(item).transpose(); 
          //iGradsAcc.row(item) = iGradsAcc.row(item)*params.rhoRMS
          //  + (1.0 - params.rhoRMS)*(tempGrad.cwiseProduct(tempGrad).transpose());
          //update rmsprop
          //for (int k = 0; k < facDim; k++) {
          //  V(item, k) -= (learnRate/(sqrt(iGradsAcc(item, k) + 0.0000001)))*tempGrad[k];
          //}
          V.row(item) -= learnRate*(tempGrad.transpose());
          
          //update item bias
          iBias(item) -= learnRate*((2.0*(r_us_est - r_us)/items.size()) 
              + 2.0*iReg*iBias(item));
        }

      }
    }    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj)) {
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



