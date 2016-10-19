#include "ModelMaxMin.h"


float ModelMaxMin::estItemRating(int user, int item) {
  bool uFound = false, iFound = true;
  float rating = 0;
  if (trainUsers.find(user) != trainUsers.end() && 
      invalidUsers.find(user) == invalidUsers.end()) {
    //found in train and not in invalid
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


float ModelMaxMin::estSetRating(int user, std::vector<int>& items) {
 
  float r_us = 0; 

  int setSz = items.size();
  float maxRat, minRat, rating;

  std::vector<float> preds(setSz, 0);
  
  rating = estItemRating(user, items[0]);
  r_us += rating;
  minRat = rating;
  maxRat = rating;

  //get predictions
  for (int i = 1; i < setSz; i++) {
    rating = estItemRating(user, items[i]);
    r_us += rating;
    if (minRat > rating) {
      minRat = rating;
    }
    if (maxRat < rating) {
      maxRat = rating;
    }
  }
  r_us = r_us/setSz; 
  r_us += uDivWt(user)*(maxRat - minRat);
  
  return r_us;
}


float ModelMaxMin::objective(const std::vector<UserSets>& uSets, 
    gk_csr_t *mat) {
  float obj = Model::objective(uSets, mat);
  for (auto&& user: trainUsers) {
    obj += uDivWt(user)*uDivWt(user)*uSetBiasReg;
  }
  return obj;
} 


 float ModelMaxMin::objective(const std::vector<UserSets>& uSets) {
  float obj = Model::objective(uSets);
  for (auto&& user: trainUsers) {
    obj += uDivWt(user)*uDivWt(user)*uSetBiasReg;
  }
  return obj;
}


void ModelMaxMin::train(const Data& data, const Params& params, Model& bestModel) {
  
  std::cout << "ModelMaxMin::train" << std::endl;

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
        UserSets uSet = data.trainSets[uInd];
        int user = uSet.user;
              
        if (uSet.itemSets.size() == 0) {
          std::cerr << "!! zero size user itemset foundi !! " << user << std::endl; 
          continue;
        }
        //select a set at random
        int setInd = dist(mt) % uSet.itemSets.size();
        auto items = uSet.itemSets[setInd].first;
        const float r_us = uSet.itemSets[setInd].second;
        const int setSz = items.size();

        if (setSz == 0) {
          std::cerr << "!! zero size itemset foundi !!" << std::endl; 
          continue;
        }
  
        //estimate rating on the set
        sumItemFactors = V.row(items[0]);    
        float rating = U.row(user).dot(V.row(items[0]));
        float r_us_est = rating;
        float minRat = rating, maxRat = rating;
        int minInd = 0, maxInd = 0;
        //compute sum of item latent factors
        for (int i = 1; i < setSz; i++) {
          rating = U.row(user).dot(V.row(items[i]));
          r_us_est += rating;
          sumItemFactors += V.row(items[i]);
          if (minRat > rating) {
            minRat = rating;
            minInd = i;
          }
          if (maxRat < rating) {
            maxRat = rating;
            maxInd = i;
          }
        }
        r_us_est = r_us_est/setSz;
        r_us_est += uDivWt(user)*(maxRat - minRat);

        //user gradient
        grad = sumItemFactors/setSz;
        grad += uDivWt(user)*(V.row(maxInd).transpose() - V.row(minInd).transpose()); 
        grad *= 2.0*(r_us_est - r_us);
        grad += 2.0*uReg*U.row(user).transpose();
        //update user
        U.row(user) -= learnRate*(grad.transpose());

        //update items
        grad = 2.0*(r_us_est - r_us)*U.row(user);
        for (int i = 0; i < setSz; i++) {
          if (i == minInd) {
            tempGrad = grad*((1.0/setSz) - uDivWt(user));
          } else if (i == maxInd) {
            tempGrad = grad*((1.0/setSz) + uDivWt(user));
          } else {
            tempGrad =  grad*(1.0/setSz);
          }
          tempGrad += 2.0*iReg*V.row(items[i]).transpose(); 
          V.row(items[i]) -= learnRate*(tempGrad.transpose());
        }

        //update w_u
        uDivWt(user) -= learnRate*((maxRat - minRat) + 2.0*uSetBiasReg*uDivWt(user));

      }
    }    
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE)) {
        break;
      }
      if (iter % 100 == 0  || iter == params.maxIter-1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << " bIter: " << bestIter
          << std::endl;
      }
    }

  }

}


