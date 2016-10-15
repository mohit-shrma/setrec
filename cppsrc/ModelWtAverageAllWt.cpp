#include "ModelWtAverageAllRange.h"

float ModelWtAverageAllRange::estItemRating(int user, int item) {
  bool uFound = false, iFound = true;
  float rating = 0;
  if (trainUsers.find(user) != trainUsers.end() && 
      invalidUsers.find(user) == invalidUsers.end()) {
    //found in train and not in invalid
    uFound = true;
    rating += uBias(user);
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


float ModelWtAverageAllRange::estSetRating(int user, std::vector<int>& items) {
  
  float r_us = 0; 

  int setSz = items.size();

  std::vector<float> preds(setSz, 0);
  std::vector<float> cumSumPreds(setSz, 0);
  //get predictions
  for (int i = 0; i < setSz; i++) {
    int item = items[i];
    preds[i] = estItemRating(user, item);
  }
  //sort predictions
  std::sort(preds.begin(), preds.end());

  cumSumPreds[0] = preds[0];
  for (int i = 1; i < setSz; i++) {
    cumSumPreds[i] = cumSumPreds[i-1] + preds[i];
  }

  //accumulate sums starting from beginning
  for (int i = 0; i < setSz; i++) {
    r_us += UWts(user, i)*(cumSumPreds[i]/(i+1));
  }

  //accumulate sums from end
  for (int i = 0; i < setSz-1; i++) {
    r_us += UWts(user, setSz + i)*((cumSumPreds[setSz-1]-cumSumPreds[i])/(setSz-(i+1)));
  }

  return r_us;
}


float ModelWtAverageAllRange::estSetRating(int user, std::vector<int>& items, 
    std::vector<Eigen::VectorXf>& cumSumItemFactors, 
    std::vector<std::pair<int, float>>& setItemRatings,
    std::vector<float>& cumSumPreds ) {
  
  float r_us = 0; 

  int setSz = items.size();

  //get predictions
  for (int i = 0; i < setSz; i++) {
    int item = items[i];
    setItemRatings[0] = std::make_pair(item, estItemRating(user, item));
  }
  //sort predictions
  std::sort(setItemRatings.begin(), setItemRatings.end(), ascComp);

  cumSumPreds[0] = itemRatings[0].second;
  cumSumItemFactors[0] = V.row(itemRatings[0].first); 
  for (int i = 1; i < setSz; i++) {
    cumSumPreds[i] = cumSumPreds[i-1] + itemRatings[i].second;
    cumSumItemFactors[i] = cumSumItemFactors[i-1] + V.row(itemRatings[i].first)
  }

  //accumulate sums starting from beginning
  for (int i = 0; i < setSz; i++) {
    r_us += UWts(user, i)*(cumSumPreds[i]/(i+1));
  }

  //accumulate sums from end
  for (int i = 0; i < setSz-1; i++) {
    r_us += UWts(user, setSz + i)*((cumSumPreds[setSz-1]-cumSumPreds[i])/(setSz-(i+1)));
  }

  return r_us;
}


void ModelWtAverageAllRange::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelWtAverageAllRange::train" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  
  std::vector<std::pair<int, float>> setItemRatings(SET_SZ);
  std::vector<Eigen::VectorXf> cumSumItemFactors(SET_SZ); 
  for (int i = 0; i < SET_SZ; i++) {
    cumSumItemFactors[i] = Eigen::VectorXf(facDim);
  }
  std::vector<float> cumSumPreds(SET_SZ);

  Eigen::VectorXf sumAllItemFactors(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
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

  trainUsers = data.trainUsers;
  trainItems = data.trainItems;
  std::cout << "train Users: " << trainUsers.size() 
    << " trainItems: " << trainItems.size() << std::endl;

  //initialize random engine
  std::mt19937 mt(params.seed);

  int maxNumSets = 0;
  for (auto&& uInd: uInds) {
    const UserSets& uSet = data.trainSets[uInd];
    if (maxNumSets < uSet.itemSets.size()) {
      maxNumSets = uSet.itemSets.size();
    }
  }
  std::cout << "Max num sets: " << maxNumSets << std::endl;
  std::uniform_int_distribution<int> dist(0, maxNumSets);

  for (int iter = 0; iter < params.maxIter; iter++) {

    std::shuffle(uInds.begin(), uInds.end(), mt);
    
    for (int subIter = 0; subIter < data.nTrainSets/nTrUsers; subIter++) {
      
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
          std::cerr << "!! zero size itemset found !!" << std::endl; 
          continue;
        }
        
        int setSz = items.size();
        
        float r_us_est = estSetRating(user, items, cumSumItemFactors, 
            setItemRatings, cumSumPreds);

        //user gradient
        //accumulated weighted item factors' sum from beginning
        grad.fill(0);
        float sumWt = 0;
        for (int i = 0; i < SET_SZ; i++) {
          grad += UWts(user, i)*cumSumItemFactors[i]/(i+1);
          sumWt += UWts(user, i);
        }
        
        //accumulate weighted item factors' sum from end
        for (int i = 0; i < SET_SZ; i++) {
          grad += UWts(user, SET_SZ + i)*(cumSumItemFactors[SET_SZ-1] -
              cumSumItemFactors[i])/(SET_SZ - (i + 1));
          sumWt += UWts(user, SET_SZ + i);
        }
        
        grad *= 2.0*(r_us_est - r_us);
        grad += 2.0*uReg*U.row(user).transpose();

        //update user
        U.row(user) -= learnRate*(grad.transpose());
        
        //update user bias
        uBias(user) -= learnRate*((2.0*(r_us_est - r_us)*sumWt) + 2.0*uBiasReg*uBias(user));

        //update items
        float iBiasGrad, sumWt;
        grad = 2.0*(r_us_est - r_us)*U.row(user);
        for (int i = 0; i < setSz; i++) {
          int item = setItemRatings[i].first;
          tempGrad = 2.0*iReg*V.row(item).transpose();
          sumWt = 0;
          int div = 0;
          //TODO: simplify below 
          for (int j = i; j < i+5; j++) {
            if (j < 5) {
              div = j + 1;
              sumWt += UWts(user, j)/div;
            } else {
              div--;
              sumWt += UWts(user, j)/div;
            }
          }
          
          tempGrad += grad*sumWt;
          iBiasGrad = sumWt;

          //update item factor
          V.row(item) -= learnRate*(tempGrad.transpose());
          //update item bias
          iBias(item) -= learnRate*(2.0*(r_us_est - r_us)*iBiasGrad
                                    + 2.0*iBiasReg*iBias(item));
        }

      }
      
//#pragma omp parallel for
      for (int uInd = 0; uInd < data.trainSets.size(); uInd++) {
        const UserSets& uSet = data.trainSets[uInd];
        int user = uSet.user;
        int nSets = uSet.itemSets.size();
        Eigen::MatrixXf Q(nSets, nWts);
        Eigen::VectorXf c(nSets);
        for (int i = 0; i < nSets; i++) {
          auto&& itemsSetNRating = uSet.itemSets[i];
          estSetRating(user, itemsSetNRating.first, cumSumItemFactors, 
              setItemRatings, cumSumPreds);
          for (int j = 0; j < SET_SZ; j++) {
            Q(i, j) = cumSumPreds[j]/(j+1);
          }
          for (int j = 0; j < SET_SZ-1; j++) {
            Q(i, SET_SZ+j) = (cumSumPreds[SET_SZ-1]-cumSumPreds[j])/(SET_SZ - (j+1));
          }
          c(i)    = itemsSetNRating.second;
        }
        
        Eigen::MatrixXf A = Q.transpose()*Q;
        Eigen::VectorXf q = - (Q.transpose()*c);
              
        //opFile << "User: " << user << " A (" << A.rows() << "," << A.cols() << ") norm: " << A.norm() << std::endl;
        //opFile << "User: " << user << " q (" << q.size() << ") norm: " << q.norm() << std::endl;

        //solve 0.5*x^T*A*x + q^T*x, s.t. 0<=x<=1 sum(x_i) = 1
        alglib::real_2d_array alg_A;
        alg_A.setlength(nWts, nWts);
        for (int i = 0; i < nWts; i++) {
          for (int j = 0; j < nWts; j++) {
            alg_A[i][j] = A(i, j);
          }
        }
        
        alglib::real_1d_array alg_q;
        alg_q.setlength(nWts);
        for (int i = 0; i < nWts; i++) {
          alg_q[i] = q(i);
        }
       
        //lower and upper bounds
        alglib::real_1d_array bndl = "[0,0,0,    0,0,0,    0,0,0]";
        alglib::real_1d_array bndu = "[10,10,10, 10,10,10, 10,10,10]";
        
        //starting point
        alglib::real_1d_array x0;
        x0.setlength(nWts);
        for (int i = 0; i < nWts; i++) {
          x0[i] = UWts(user, i);
        }
        
        //constraint: x0 + x1 + x2 + ... + x8 = 1;
        alglib::real_2d_array constr = "[[1,1,1, 1,1,1, 1,1,1, 1]]";
        alglib::integer_1d_array ct = "[0]"; // = constraint 

        //solution
        alglib::real_1d_array x;

        alglib::minqpstate state;
        alglib::minqpreport rep;
      
        //scaling parameter indicating that model variables are in same scale
        alglib::real_1d_array s = "[1,1,1, 1,1,1, 1,1,1]";

        //create solver
        minqpcreate(nWts, state);
        minqpsetquadraticterm(state, alg_A);
        minqpsetlinearterm(state, alg_q);
        minqpsetbc(state, bndl, bndu);
        minqpsetlc(state, constr, ct);
        //minqpsetstartingpoint(state, x0);
        
        //set scale of the variables
        minqpsetscale(state, s);
        
        //std::cout << "Solving for user: " << user << std::endl;

        //solve problem with BLEIC-based QP solver
        minqpsetalgobleic(state, 0.0, 0.0, 0.0, 0);
        minqpoptimize(state);
        minqpresults(state, x, rep);
        
        bool isFail = int(rep.terminationtype) < 0;
        if (!isFail) {
          for (int i = 0; i < nWts; i++) {
            UWts(user, i) = x[i];
          }
        } else {
          std::cout << "Failed for user: " << user << int(rep.terminationtype) 
            << std::endl;
        }
        
      }
    
    }
      
    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE)) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      
      if (iter%10 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE:" << bestValRMSE 
          << " train RMSE:" << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << std::endl;
      }

    }

  }

}


