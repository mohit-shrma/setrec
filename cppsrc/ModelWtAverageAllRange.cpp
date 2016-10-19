#include "ModelWtAverageAllRange.h"

bool ModelWtAverageAllRange::isTerminateModelWPartIRMSE(Model& bestModel, 
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
   
    if (iter - bestIter >= CHANCE_ITER/2) {
      if (learnRate > 5e-4) {
        std::cout << "Changing learn rate from: " << learnRate;
        learnRate = learnRate/2;
        std::cout << " to: " << learnRate << std::endl; 
      }
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


float ModelWtAverageAllRange::estItemRating(int user, int item) {
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


float ModelWtAverageAllRange::estSetRating(int user, std::vector<int>& items) {
 
  float r_us = 0; 

  int setSz = items.size();

  std::vector<float> preds(setSz, 0);
  //get predictions
  for (int i = 0; i < setSz; i++) {
    int item = items[i];
    preds[i] = estItemRating(user, item);
  }
  //sort predictions
  std::sort(preds.begin(), preds.end());

  float cumSum = 0;
  for (int i = 0; i < setSz; i++) {
    cumSum += preds[i];
    r_us += UWts(user, i)*cumSum/(i+1);
  }

  //accumulate sums from end
  for (int i = 0; i < setSz-1; i++) {
    cumSum -= preds[i];
    r_us += UWts(user, setSz + i)*cumSum/(setSz-(i+1));
  }

  return r_us;
}


void ModelWtAverageAllRange::estSetRatings(int user, const std::vector<int>& items,
    std::vector<float>& setRatings) {
 
  int setSz = items.size();

  std::vector<float> preds(setSz, 0);
  //get predictions
  for (int i = 0; i < setSz; i++) {
    int item = items[i];
    preds[i] = estItemRating(user, item);
  }
  //sort predictions
  std::sort(preds.begin(), preds.end());

  float cumSum = 0;
  int wtInd = 0;
  for (int i = 0; i < setSz; i++) {
    cumSum += preds[i];
    setRatings[wtInd++] = cumSum/(i+1);
  }

  //accumulate sums from end
  for (int i = 0; i < setSz-1; i++) {
    cumSum -= preds[i];
    setRatings[wtInd++] = cumSum/(setSz-(i+1));
  }

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

  Eigen::VectorXf cumFac(facDim);
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
  std::cout << "Objective: " << objective(data.trainSets);
  std::cout << " train sets RMSE:" << rmse(data.trainSets) 
            << " test sets RMSE:" << rmse(data.testSets) 
            << " train ratings RMSE: " << rmse(data.partTrainMat) 
            << " test ratings RMSE: " << rmse(data.partTestMat) 
            << std::endl; 

  auto partUIRatingsTup = getUIRatingsTup(data.partTrainMat);
  
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
        const auto& items = uSet.itemSets[setInd].first;
        const float r_us = uSet.itemSets[setInd].second;

        if (items.size() == 0) {
          std::cerr << "!! zero size itemset found !!" << std::endl; 
          continue;
        }
        
        int setSz = items.size();
        
        //get predictions
        for (int i = 0; i < setSz; i++) {
          int item = items[i];
          setItemRatings[i] = std::make_pair(item, estItemRating(user, item));
        }
        //sort predictions
        std::sort(setItemRatings.begin(), setItemRatings.end(), ascComp);
        
        float r_us_est = 0;
        float cumSum = 0;
        float sumWt = 0;
        cumFac.fill(0);
        grad.fill(0);
        //accumulate sums from beginning
        for (int i = 0; i < setSz; i++) {
          cumSum += setItemRatings[i].second;
          cumFac += V.row(setItemRatings[i].first);
          r_us_est += UWts(user, i)*cumSum/(i+1);
          grad     += UWts(user, i)*cumFac/(i+1);
          sumWt    += UWts(user, i);
        }

        //accumulate sums from end
        for (int i = 0; i < setSz-1; i++) {
          cumSum -= setItemRatings[i].second;
          cumFac -= V.row(setItemRatings[i].first);
          r_us_est += UWts(user, setSz + i)*cumSum/(setSz-(i+1));
          grad     += UWts(user, setSz + i)*cumFac/(setSz-(i+1));
          sumWt    += UWts(user, setSz + i);
        }

        
        grad *= 2.0*(r_us_est - r_us);
        grad += 2.0*uReg*U.row(user).transpose();

        //update user
        U.row(user) -= learnRate*(grad.transpose());
        
        //update user bias
        //uBias(user) -= learnRate*((2.0*(r_us_est - r_us)*sumWt) + 2.0*uBiasReg*uBias(user));

        //update items
        float iBiasGrad;
        grad = 2.0*(r_us_est - r_us)*U.row(user);
        for (int i = 0; i < setSz; i++) {
          int item = setItemRatings[i].first;
          tempGrad = 2.0*iReg*V.row(item).transpose();
          sumWt = 0;
          int buckSz = 0;
          for (int j = i; j < i+5; j++) {
            if (j < 5) {
              buckSz = j + 1;
              sumWt += UWts(user, j)/buckSz;
            } else {
              buckSz--;
              sumWt += UWts(user, j)/buckSz;
            }
          }
          
          tempGrad += grad*sumWt;
          iBiasGrad = sumWt;

          //update item factor
          V.row(item) -= learnRate*(tempGrad.transpose());
          //update item bias
          //iBias(item) -= learnRate*(2.0*(r_us_est - r_us)*iBiasGrad
          //                          + 2.0*iBiasReg*iBias(item));
        }
        
      }
    }
    
    /*
    std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
    for (auto&& uiRating: partUIRatingsTup) {
      int user       = std::get<0>(uiRating);
      int item       = std::get<1>(uiRating);
      float r_ui     = std::get<2>(uiRating);
      float r_ui_est = estItemRating(user, item);
      float diff     = r_ui_est - r_ui;
      //update user latent factor
      U.row(user) -= learnRate*(2.0*diff*V.row(item) + 2.0*uReg*U.row(user));
      //update item latent factor
      V.row(item) -= learnRate*(2.0*diff*U.row(user) + 2.0*iReg*V.row(item));
    }
    */

    if (false) {
      //std::cout << "B4 QP Objective: " << objective(data.trainSets) << std::endl;
#pragma omp parallel for
      for (int uInd = 0; uInd < data.trainSets.size(); uInd++) {
        const UserSets& uSet = data.trainSets[uInd];
        int user = uSet.user;
        int nSets = uSet.itemSets.size();
        Eigen::MatrixXf Q(nSets, nWts);
        Eigen::VectorXf c(nSets);
        for (int k = 0; k < nSets; k++) {
          auto&& itemsSetNRating = uSet.itemSets[k];
          const auto& items = itemsSetNRating.first;
          int setSz = items.size();
          
          std::vector<float> preds(setSz, 0);
          //get predictions
          for (int i = 0; i < setSz; i++) {
            int item = items[i];
            preds[i] = estItemRating(user, item);
          }
          //sort predictions
          std::sort(preds.begin(), preds.end());
          
          float cumSum = 0;
          for (int j = 0; j < setSz; j++) {
            cumSum += preds[j];
            Q(k, j) = cumSum/(j+1);
          }
         
          //accumulate sums from end
          for (int j = 0; j < setSz-1; j++) {
            cumSum -= preds[j];
            Q(k, setSz + j) = cumSum/(setSz - (j+1));
          }
          
          c(k)    = itemsSetNRating.second;
        }
        
              
        //std::cout << "User: " << user << " A (" << A.rows() << "," << A.cols() << ") norm: " << A.norm() << std::endl;
        //std::cout << "User: " << user << " q (" << q.size() << ") norm: " << q.norm() << std::endl;

        //solve 0.5*x^T*A*x + q^T*x, s.t. 0<=x<=1 sum(x_i) = 1
        alglib::real_2d_array alg_A;
        alg_A.setlength(nWts, nWts);
        for (int i = 0; i < nWts; i++) {
          for (int j = i; j < nWts; j++) {
            alg_A[i][j] = Q.col(i).dot(Q.col(j));
            alg_A[j][i] = alg_A[i][j];
          }
        }

        alglib::real_1d_array alg_q;
        alg_q.setlength(nWts);
        for (int i = 0; i < nWts; i++) {
          alg_q[i] = -(Q.col(i).dot(c));
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
          /*
          int maxInd = 0;
          float maxX = x[maxInd];
          for (int i = 1; i < nWts; i++) {
            if (x[i] > maxX) {
              maxInd = i;
              maxX = x[i];
            }
          }
          */

          for (int i = 0; i < nWts; i++) {
            UWts(user, i) = x[i];
            //UWts(user, i) = 0;
          }
          //UWts(user, maxInd) = 1;

        } else {
          std::cout << "Failed for user: " << user << int(rep.terminationtype) 
            << std::endl;
        }
        
      }
    }

    if (false) {
      
      //std::cout << "B4 QP Objective: " << objective(data.trainSets) << std::endl;
#pragma omp parallel for
      for (int uInd = 0; uInd < data.trainSets.size(); uInd++) {
        const UserSets& uSet = data.trainSets[uInd];
        int user = uSet.user;
        int nSets = uSet.itemSets.size();
        
        std::vector<float> setRatings(nWts, 0);
        std::vector<float> wtRMSE(nWts, 0);
        float diff;
        for (int k = 0; k < nSets; k++) {
          auto&& itemsSetNRating = uSet.itemSets[k];
          const auto& items = itemsSetNRating.first;
          float r_us = itemsSetNRating.second;
          estSetRatings(user, items, setRatings);
          for (int i = 0; i < nWts; i++) {
            diff = setRatings[i] - r_us;
            wtRMSE[i] += diff*diff;
          }
        }
        
        int minInd = 0;
        for (int i = 0; i < nWts; i++) {
          wtRMSE[i] = std::sqrt(wtRMSE[i]/nSets);
          if (wtRMSE[minInd] < wtRMSE[i]) {
            minInd = i;
          } 
        }
        UWts.row(user).fill(0);
        UWts(user, minInd) = 1;
      }
    
    }


    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      if (isTerminateModel(bestModel, data, iter, bestIter, bestObj, prevObj,
            bestValRMSE, prevValRMSE)) {
      //if (isTerminateModelWPartIRMSE(bestModel, data, iter, bestIter, bestObj, prevObj,
      //      bestValRMSE, prevValRMSE)) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      
      if (iter%50 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val RMSE: " 
          << prevValRMSE << " best val RMSE: " << bestValRMSE 
          << " train RMSE: " << rmse(data.trainSets) 
          << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
          << " bIter: " << bestIter
          << std::endl;
      }

    }

  }

  /* 
  std::ofstream opFile("user_weights_qp_outer1.txt");
  for (int u = 0; u < nUsers; u++) {
  //for (auto&& u: trainUsers) {
    //opFile << u << " ";
    for (int i = 0; i < nWts; i++) {
      opFile << UWts(u, i) << " "; 
      if (i == 3 || i == 4) {
        //opFile << "|";
      }
    }
    opFile << std::endl;
  }
  opFile.close();
  */ 
}


