#include "ModelWtAverageAllRange.h"


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


float ModelWtAverageAllRange::estUSetsRMSE(const UserSets& uSet, alglib::real_1d_array& wts) {
  int user = uSet.user;
  std::vector<float> setRatings(nWts, 0);
  float diff = 0;
  for (const auto& itemSet: uSet.itemSets) {
    const auto& items = itemSet.first;
    float r_us = itemSet.second;
    float r_us_est = 0;
    estSetRatings(user, items, setRatings);
    for (int i = 0; i < nWts; i++) {
      r_us_est += wts[i]*setRatings[i];
    }
    diff += (r_us_est - r_us)*(r_us_est - r_us);
  }
  return std::sqrt(diff/uSet.itemSets.size());
} 


float ModelWtAverageAllRange::estUSetsRMSE(Eigen::MatrixXf& Q, Eigen::VectorXf& c, 
    alglib::real_1d_array& wts) {
  std::vector<float> setRatings(nWts, 0);
  int nSets = Q.rows();
  float diff = 0, r_us_est = 0;
  for (int i = 0; i < nSets; i++) {
    r_us_est = 0;
    for (int j = 0; j < nWts; j++) {
      r_us_est += wts[j]*Q(i, j);
    }
    diff += (r_us_est - c(i))*(r_us_est - c(i));
  }
  return std::sqrt(diff/nSets);
} 


float ModelWtAverageAllRange::estSetRating(int user, const std::vector<int>& items, int exSetInd) {
 
  int setSz = items.size();
  bool isExSetComputed = false;
  float exSetRating;
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
    if (exSetInd == wtInd) {
      exSetRating = cumSum/(i+1);
      isExSetComputed = true;
      break;
    }
    wtInd++;
    //setRatings[wtInd++] = cumSum/(i+1);
  }

  //accumulate sums from end
  for (int i = 0; i < setSz-1 && !isExSetComputed; i++) {
    cumSum -= preds[i];
    if (exSetInd == wtInd) {
      exSetRating = cumSum/(setSz-(i+1));
      isExSetComputed = true;
      break;
    }
    wtInd++;
    //setRatings[wtInd++] = cumSum/(setSz-(i+1));
  }
  
  if (!isExSetComputed) {
    std::cerr << "Extremal set: "<< exSetInd << " not found" << std::endl;
  }

  return exSetRating;
}


float ModelWtAverageAllRange::estUExSetRMSE(const UserSets& uSet, int exSetInd) {
  float diff = 0;
  for (const auto& itemSet: uSet.itemSets) {
    const auto& items = itemSet.first;
    const float r_us = itemSet.second;
    float r_us_est = estSetRating(uSet.user, items, exSetInd);
    diff += (r_us_est - r_us)*(r_us_est - r_us);
  }
  return std::sqrt(diff/uSet.itemSets.size());
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

  Eigen::MatrixXf UGradAvg(nUsers, facDim);
  Eigen::MatrixXf VGradAvg(nItems, facDim);
  Eigen::MatrixXf UGradSqAvg(nUsers, facDim);
  Eigen::MatrixXf VGradSqAvg(nItems, facDim);

  UGradAvg.fill(0); VGradAvg.fill(0);
  UGradSqAvg.fill(0); VGradSqAvg.fill(0);
 
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

  //UWts init with GT
  //readEigenMat("userPickiness_syn_1.txt", UWts, nUsers, 9);

  //UWts init with avg  
  /*
  UWts.fill(0);
  for (int u = 0; u < nUsers; u++) {
    UWts(u, 4)  = 1;  
  }
  */ 

  int nUserCh = 0;

  for (int iter = 0; iter < params.maxIter; iter++) {

    std::shuffle(uInds.begin(), uInds.end(), mt);
   
    nUserCh = 0;

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

    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg);
    }

    if (false) {
      //std::cout << "B4 QP Objective: " << objective(data.trainSets) << std::endl;
#pragma omp parallel for reduction(+:nUserCh)
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
        bool isCh = false;
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
            if ( fabs(UWts(user, i) - x[i]) > EPS) {
              isCh = true;
            }
            UWts(user, i) = x[i];
            //UWts(user, i) = 0;
          }
          //UWts(user, maxInd) = 1;

        } else {
          std::cout << "Failed for user: " << user << int(rep.terminationtype) 
            << std::endl;
        }

        if (isCh) {
          nUserCh++;
        }
        
      }
    }


    //if (iter % 1 == 0) {
    if (true) {
      //std::cout << "B4 greedy Objective: " << objective(data.trainSets) << std::endl;
#pragma omp parallel for reduction(+: nUserCh)
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
        int oldInd = 0;
        for (int i = 0; i < nWts; i++) {
          wtRMSE[i] = std::sqrt(wtRMSE[i]/nSets);
          if (wtRMSE[minInd] > wtRMSE[i]) {
            minInd = i;
          } 
          if (UWts(user, i) > 0) {
            oldInd = i;
          }
        }

        UWts.row(user).fill(0);
        UWts(user, minInd) = 1;

        if (oldInd != minInd) {
          nUserCh++;
        }
      }
    }

    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      
      if ((!params.isMixRat && isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE))) {
        break;
        //save best model
        //bestModel.save(params.prefix);
      } else if ((params.isMixRat && isTerminateModelWPartIRMSE(bestModel, data, iter, 
            bestIter, bestObj, prevObj, bestValRMSE, prevValRMSE))) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      
      /*
      if (isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE)) {
        break;
      }
      */
      if (iter%250 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val: " 
          << prevValRMSE << " best val: " << bestValRMSE 
          << " train: " << rmse(data.trainSets) 
          << " train ratings: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings: " << rmse(data.testSets, data.ratMat)
          << " nUserCh: " << nUserCh
          << " bIter: " << bestIter
          << std::endl;
      }

    }

  }
  
  /*
  float hits = 0, count = 0, diff = 0, avgExSetRMSE = 0, avgEstExSetRMSE = 0; 
  float avgUNNZ = 0;
  std::ofstream opFile("user_weights_esqp.txt");
  
  for (const auto& userSets: data.trainSets) {
    
    int user = userSets.user;
    auto exSetNRMSE = userSets.getTopExtremalSubsetWRMSE(data.ratMat);
    int exSetInd = exSetNRMSE.first;
    float exSetRMSE = exSetNRMSE.second;
    avgExSetRMSE += exSetRMSE;

    std::vector<size_t> idx(2*SET_SZ-1);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), 
        [=] (size_t i1, size_t i2) { return UWts(user, i1) > UWts(user, i2); });

    int maxWtInd = 0;
    int nnz = 0;
    if (UWts(user, 0) != 0) { nnz++; }
    for (int i = 1; i < 2*SET_SZ-1; i++) {
      if (UWts(user, i) > UWts(user, maxWtInd)) {
        maxWtInd = i;
      }
      if (UWts(user, i) != 0 ) { nnz++; }
    }
    avgUNNZ += nnz;

    int isHit = 0;
    //if (maxWtInd == exSetInd) {
    if (idx[0] == exSetInd) { // || idx[1] == exSetInd || idx[2] == exSetInd) {
       hits += 1;
       isHit = 1;
    }
    count += 1;

    float estExSetRMSE = estUExSetRMSE(userSets, maxWtInd);
    avgEstExSetRMSE += estExSetRMSE;
    
    diff += fabs(exSetRMSE - estExSetRMSE);
    opFile << user << " " << exSetInd << " " << maxWtInd << " " 
      << exSetRMSE << " " << estExSetRMSE << " " << userSets.itemSets.size()
      << " " << nnz  << std::endl;
  }
  opFile.close();
  std::cout << "Fraction of user hits: " << hits/count << std::endl;
  std::cout << "Avg diff b/w orig & est exSet: " << diff/count << std::endl;
  std::cout << "avgExSetRMSE: " << avgExSetRMSE/count 
    << " avgEstExSetRMSE: " << avgEstExSetRMSE/count << std::endl;
  std::cout << "avgNNZCoeff: " << avgUNNZ/count << std::endl;
  */ 
}


void ModelWtAverageAllRange::trainQP(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelWtAverageAllRange::trainQP" << std::endl; 
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

  Eigen::MatrixXf UGradAvg(nUsers, facDim);
  Eigen::MatrixXf VGradAvg(nItems, facDim);
  Eigen::MatrixXf UGradSqAvg(nUsers, facDim);
  Eigen::MatrixXf VGradSqAvg(nItems, facDim);

  UGradAvg.fill(0); VGradAvg.fill(0);
  UGradSqAvg.fill(0); VGradSqAvg.fill(0);
 
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

  //UWts init with GT
  //readEigenMat("userPickiness_syn_1.txt", UWts, nUsers, 9);

  //UWts init with avg  
  /*
  UWts.fill(0);
  for (int u = 0; u < nUsers; u++) {
    UWts(u, 4)  = 1;  
  }
  */ 

  int nUserCh = 0;

  for (int iter = 0; iter < params.maxIter; iter++) {

    std::shuffle(uInds.begin(), uInds.end(), mt);
   
    nUserCh = 0;

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

    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg);
    }

    if (true) {
      //std::cout << "B4 QP Objective: " << objective(data.trainSets) << std::endl;
#pragma omp parallel for reduction(+:nUserCh)
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
        bool isCh = false;
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
            if ( fabs(UWts(user, i) - x[i]) > EPS) {
              isCh = true;
            }
            UWts(user, i) = x[i];
            //UWts(user, i) = 0;
          }
          //UWts(user, maxInd) = 1;

        } else {
          std::cout << "Failed for user: " << user << int(rep.terminationtype) 
            << std::endl;
        }

        if (isCh) {
          nUserCh++;
        }
        
      }
    }

    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      
      if ((!params.isMixRat && isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE))) {
        break;
        //save best model
        //bestModel.save(params.prefix);
      } else if ((params.isMixRat && isTerminateModelWPartIRMSE(bestModel, data, iter, 
            bestIter, bestObj, prevObj, bestValRMSE, prevValRMSE))) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      
      /*
      if (isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE)) {
        break;
      }
      */
      if (iter%250 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter 
          << " obj:" << prevObj << " best Obj: " << bestObj  
          << " val: " << prevValRMSE << " best val: " << bestValRMSE 
          << " train: " << rmse(data.trainSets) 
          << " train ratings: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings: " << rmse(data.testSets, data.ratMat)
          << " nUserCh: " << nUserCh
          << " bIter: " << bestIter
          << std::endl;
      }

    }

  }
  
  /*
  float hits = 0, count = 0, diff = 0, avgExSetRMSE = 0, avgEstExSetRMSE = 0; 
  float avgUNNZ = 0;
  std::ofstream opFile("user_weights_esqp.txt");
  
  for (const auto& userSets: data.trainSets) {
    
    int user = userSets.user;
    auto exSetNRMSE = userSets.getTopExtremalSubsetWRMSE(data.ratMat);
    int exSetInd = exSetNRMSE.first;
    float exSetRMSE = exSetNRMSE.second;
    avgExSetRMSE += exSetRMSE;

    std::vector<size_t> idx(2*SET_SZ-1);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), 
        [=] (size_t i1, size_t i2) { return UWts(user, i1) > UWts(user, i2); });

    int maxWtInd = 0;
    int nnz = 0;
    if (UWts(user, 0) != 0) { nnz++; }
    for (int i = 1; i < 2*SET_SZ-1; i++) {
      if (UWts(user, i) > UWts(user, maxWtInd)) {
        maxWtInd = i;
      }
      if (UWts(user, i) != 0 ) { nnz++; }
    }
    avgUNNZ += nnz;

    int isHit = 0;
    //if (maxWtInd == exSetInd) {
    if (idx[0] == exSetInd) { // || idx[1] == exSetInd || idx[2] == exSetInd) {
       hits += 1;
       isHit = 1;
    }
    count += 1;

    float estExSetRMSE = estUExSetRMSE(userSets, maxWtInd);
    avgEstExSetRMSE += estExSetRMSE;
    
    diff += fabs(exSetRMSE - estExSetRMSE);
    opFile << user << " " << exSetInd << " " << maxWtInd << " " 
      << exSetRMSE << " " << estExSetRMSE << " " << userSets.itemSets.size()
      << " " << nnz  << std::endl;
  }
  opFile.close();
  std::cout << "Fraction of user hits: " << hits/count << std::endl;
  std::cout << "Avg diff b/w orig & est exSet: " << diff/count << std::endl;
  std::cout << "avgExSetRMSE: " << avgExSetRMSE/count 
    << " avgEstExSetRMSE: " << avgEstExSetRMSE/count << std::endl;
  std::cout << "avgNNZCoeff: " << avgUNNZ/count << std::endl;
  */ 
}


void ModelWtAverageAllRange::trainQPSmooth(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelWtAverageAllRange::trainQPSmooth" << std::endl; 
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
  
  std::vector<int> uUpdCount(nUsers, 0);
  std::vector<int> itemUpdCount(nItems, 0);

  Eigen::MatrixXf UGradAvg(nUsers, facDim);
  Eigen::MatrixXf VGradAvg(nItems, facDim);
  Eigen::MatrixXf UGradSqAvg(nUsers, facDim);
  Eigen::MatrixXf VGradSqAvg(nItems, facDim);

  UGradAvg.fill(0); VGradAvg.fill(0);
  UGradSqAvg.fill(0); VGradSqAvg.fill(0);
  
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

  //UWts init with GT
  //readEigenMat("userPickiness_syn_1.txt", UWts, nUsers, 9);

  //UWts init with avg  
  /*
  UWts.fill(0);
  for (int u = 0; u < nUsers; u++) {
    UWts(u, 4)  = 1;  
  }
  */ 

  int nUserCh = 0;

  if (params.isMixRat) {
    std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
    updateFacUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg);
  }

  for (int iter = 0; iter < params.maxIter; iter++) {

    std::shuffle(uInds.begin(), uInds.end(), mt);
   
    nUserCh = 0;

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
        //U.row(user) -= learnRate*(grad.transpose());
        //ADAMUpdate(U, user, UGradAvg, UGradSqAvg, grad, 0.9, 0.999, learnRate, uUpdCount[user]);
        //uUpdCount[user]++;
        RMSPropUpdate(U, user, UGradSqAvg, grad, learnRate, 0.9);

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
          //V.row(item) -= learnRate*(tempGrad.transpose());
          //ADAMUpdate(V, item, VGradAvg, VGradSqAvg, tempGrad, 0.9, 0.999, learnRate, itemUpdCount[item]);
          //itemUpdCount[item]++;
          RMSPropUpdate(V, item, VGradSqAvg, tempGrad, learnRate, 0.9);
          //update item bias
          //iBias(item) -= learnRate*(2.0*(r_us_est - r_us)*iBiasGrad
          //                          + 2.0*iBiasReg*iBias(item));
        }
        
      }
    }

    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg);
    }

    if (true) {
      //std::cout << "B4 QP Objective: " << objective(data.trainSets) << std::endl;
#pragma omp parallel for reduction(+:nUserCh)
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
      
        //scaling parameter indicating that model variables are in same scale
        alglib::real_1d_array s = "[1,1,1, 1,1,1, 1,1,1]";
        
        //best solution
        alglib::real_1d_array xBest;
        float bestRMSE;

        bool isCh = false;
        
        for (int i = 0; i < 9; i++) {
          //constraint: x0 + x1 + x2 + ... + x8 = 1;
          alglib::real_2d_array constr; //= "[[1,1,1, 1,1,1, 1,1,1, 1]]";
          constr.setlength(1 + 8, 10); //1 equality and 8 inequality constraints
          //constr.setlength(1 + 9, 10); //1 equality and 9 inequality constraints
          for (int j = 0; j < 9; j++) {
          //for (int j = 0; j < 10; j++) {
            for (int k = 0; k < 10; k++) {
              constr[j][k] = 0;
            }
          }
          
          alglib::integer_1d_array ct;// = "[0]"; // = constraint 
          ct.setlength(9);
          //ct.setlength(10);

          //assign equality constraint
          ct[0] = 0;
          for (int j = 0; j < 10; j++) {
            constr[0][j] = 1;
          }
          
          //assign inequality constraints
          int j = 1;
          for (int k = 0; k < i; k++) {
            constr[j][k]   = 1;
            constr[j][k+1] = -1;
            constr[j][9]  = -1e-4;
            ct[j] = -1;
            j++;
          }
          for (int k = i; k < 8; k++) {
            constr[j][k] = -1;
            constr[j][k+1] = 1;
            constr[j][9]  = -1e-4;
            ct[j] = -1;
            j++;
          }
        
          //following is for explicit inequality constraint
          //constr[j][i] = -1;
          //constr[j][9] = -gamma;//-0.9;
          //ct[j++] = -1;
          

          alglib::minqpstate state;
          alglib::minqpreport rep;
          //solution
          alglib::real_1d_array x;
          //create solver
          minqpcreate(nWts, state);
          minqpsetquadraticterm(state, alg_A);
          minqpsetlinearterm(state, alg_q);
          minqpsetbc(state, bndl, bndu);
          minqpsetlc(state, constr, ct);
          //minqpsetstartingpoint(state, x0);
          
          //set scale of the variables
          minqpsetscale(state, s);
          
          //solve problem with BLEIC-based QP solver
          minqpsetalgobleic(state, 0.0, 0.0, 0.0, 0);
          minqpoptimize(state);
          minqpresults(state, x, rep);
        
          bool isFail = int(rep.terminationtype) < 0;
          if (!isFail) {
            //save best solution
            float rmse = estUSetsRMSE(Q, c, x);
            if (0 == i || bestRMSE > rmse) {
              bestRMSE = rmse;
              xBest = x;
            } 
          } else {
            std::cout << "Failed for user: " << user << int(rep.terminationtype) 
              << std::endl;
          }
  
        } 
        
        int oldMaxInd = 0;
        int newMaxInd = 0;
        for (int i = 0; i < nWts; i++) {
          
          if (UWts(user, i) > UWts(user, oldMaxInd)) {
            oldMaxInd = i;
          }
          
          if (xBest(newMaxInd) < xBest[i]) {
            newMaxInd = i;
          }

          UWts(user, i) = xBest[i];
        }
        
        if (oldMaxInd != newMaxInd) {
          isCh = true;
        }

        if (isCh) {
          nUserCh++;
        }
        
      }
    }

    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      
      if ((!params.isMixRat && isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE, false))) {
        break;
        //save best model
        //bestModel.save(params.prefix);
      } else if ((params.isMixRat && isTerminateModelWPartIRMSE(bestModel, data, iter, 
            bestIter, bestObj, prevObj, bestValRMSE, prevValRMSE))) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      
      /*
      if (isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE)) {
        break;
      }
      */
      if (iter%250 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter 
          << " obj:" << prevObj << " best Obj: " << bestObj  
          << " val: " << prevValRMSE << " best val: " << bestValRMSE 
          << " train: " << rmse(data.trainSets) 
          << " train ratings: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings: " << rmse(data.testSets, data.ratMat)
          << " nUserCh: " << nUserCh
          << " bIter: " << bestIter
          << std::endl;
      }

    }

  }
  
  
  float hits = 0, count = 0, diff = 0, avgExSetRMSE = 0, avgEstExSetRMSE = 0; 
  float avgUNNZ = 0;
  //std::ofstream opFile("user_weights_esqp_smooth.txt");
  //std::ofstream opFile2("user_weights_qp_smooth.txt");
  
  for (const auto& userSets: data.trainSets) {
    
    int user = userSets.user;
    auto exSetNRMSE = userSets.getTopExtremalSubsetWRMSE(data.ratMat);
    int exSetInd = exSetNRMSE.first;
    float exSetRMSE = exSetNRMSE.second;
    avgExSetRMSE += exSetRMSE;

    std::vector<size_t> idx(2*SET_SZ-1);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), 
        [=] (size_t i1, size_t i2) { return UWts(user, i1) > UWts(user, i2); });

    int maxWtInd = 0;
    int nnz = 0;
    if (UWts(user, 0) != 0) { nnz++; }
    for (int i = 1; i < 2*SET_SZ-1; i++) {
      if (UWts(user, i) > UWts(user, maxWtInd)) {
        maxWtInd = i;
      }
      if (UWts(user, i) != 0 ) { nnz++; }
    }
    avgUNNZ += nnz;

    int isHit = 0;
    //if (maxWtInd == exSetInd) {
    if (idx[0] == exSetInd) { // || idx[1] == exSetInd || idx[2] == exSetInd) {
       hits += 1;
       isHit = 1;
    }
    count += 1;

    float estExSetRMSE = estUExSetRMSE(userSets, maxWtInd);
    avgEstExSetRMSE += estExSetRMSE;
    
    diff += fabs(exSetRMSE - estExSetRMSE);
    /*
    opFile << user << " " << exSetInd << " " << maxWtInd << " " 
      << exSetRMSE << " " << estExSetRMSE << " " << userSets.itemSets.size()
      << " " << nnz  << std::endl;
    opFile2 << user << " ";
    for (int i = 0; i < nWts; i++) {
      opFile2 << UWts(user, i) << " ";
    }
    opFile2 << std::endl;
    */
  }
  //opFile.close();
  //opFile2.close();
  std::cout << "Fraction of user hits: " << hits/count << std::endl;
  std::cout << "Avg diff b/w orig & est exSet: " << diff/count << std::endl;
  std::cout << "avgExSetRMSE: " << avgExSetRMSE/count 
    << " avgEstExSetRMSE: " << avgEstExSetRMSE/count << std::endl;
  std::cout << "avgNNZCoeff: " << avgUNNZ/count << std::endl;
  
}


void ModelWtAverageAllRange::trainGreedy(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelWtAverageAllRange::trainGreedy" << std::endl; 
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

  Eigen::MatrixXf UGradAvg(nUsers, facDim);
  Eigen::MatrixXf VGradAvg(nItems, facDim);
  Eigen::MatrixXf UGradSqAvg(nUsers, facDim);
  Eigen::MatrixXf VGradSqAvg(nItems, facDim);
  
  UGradAvg.fill(0); VGradAvg.fill(0);
  UGradSqAvg.fill(0); VGradSqAvg.fill(0);
  
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

  //UWts init with GT
  //readEigenMat("userPickiness_syn_1.txt", UWts, nUsers, 9);

  //UWts init with avg  
  /*
  UWts.fill(0);
  for (int u = 0; u < nUsers; u++) {
    UWts(u, 4)  = 1;  
  }
  */ 

  int nUserCh = 0;

  for (int iter = 0; iter < params.maxIter; iter++) {

    std::shuffle(uInds.begin(), uInds.end(), mt);
   
    nUserCh = 0;

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
        //U.row(user) -= learnRate*(grad.transpose());
        RMSPropUpdate(U, user, UGradSqAvg, grad, learnRate, 0.9);
        
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
          //V.row(item) -= learnRate*(tempGrad.transpose());
          RMSPropUpdate(V, item, VGradSqAvg, tempGrad, learnRate, 0.9);
          //update item bias
          //iBias(item) -= learnRate*(2.0*(r_us_est - r_us)*iBiasGrad
          //                          + 2.0*iBiasReg*iBias(item));
        }
        
      }
    }

    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg);
    }

    if (true) {
      //std::cout << "B4 greedy Objective: " << objective(data.trainSets) << std::endl;
#pragma omp parallel for reduction(+: nUserCh)
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
        int oldInd = 0;
        for (int i = 0; i < nWts; i++) {
          wtRMSE[i] = std::sqrt(wtRMSE[i]/nSets);
          if (wtRMSE[minInd] > wtRMSE[i]) {
            minInd = i;
          } 
          if (UWts(user, i) > 0) {
            oldInd = i;
          }
        }

        UWts.row(user).fill(0);
        UWts(user, minInd) = 1;

        if (oldInd != minInd) {
          nUserCh++;
        }
      }
    }

    //objective check
    if (iter % OBJ_ITER == 0 || iter == params.maxIter-1) {
      
      if ((!params.isMixRat && isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE, false))) {
        break;
        //save best model
        //bestModel.save(params.prefix);
      } else if ((params.isMixRat && isTerminateModelWPartIRMSE(bestModel, data, iter, 
            bestIter, bestObj, prevObj, bestValRMSE, prevValRMSE))) {
        //save best model
        //bestModel.save(params.prefix);
        break;
      }
      
      /*
      if (isTerminateModel(bestModel, data, iter, bestIter,
            bestObj, prevObj, bestValRMSE, prevValRMSE)) {
        break;
      }
      */
      if (iter%250 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj << " val: " 
          << prevValRMSE << " best val: " << bestValRMSE 
          << " train: " << rmse(data.trainSets) 
          << " train ratings: " << rmse(data.trainSets, data.ratMat) 
          << " test ratings: " << rmse(data.testSets, data.ratMat)
          << " nUserCh: " << nUserCh
          << " bIter: " << bestIter
          << std::endl;
      }

    }

  }
  
  float hits = 0, count = 0, diff = 0, avgExSetRMSE = 0, avgEstExSetRMSE = 0; 
  float avgUNNZ = 0;
  //std::ofstream opFile("user_weights_esqp.txt");
  
  for (const auto& userSets: data.trainSets) {
    
    int user = userSets.user;
    auto exSetNRMSE = userSets.getTopExtremalSubsetWRMSE(data.ratMat);
    int exSetInd = exSetNRMSE.first;
    float exSetRMSE = exSetNRMSE.second;
    avgExSetRMSE += exSetRMSE;

    std::vector<size_t> idx(2*SET_SZ-1);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), 
        [=] (size_t i1, size_t i2) { return UWts(user, i1) > UWts(user, i2); });

    int maxWtInd = 0;
    int nnz = 0;
    if (UWts(user, 0) != 0) { nnz++; }
    for (int i = 1; i < 2*SET_SZ-1; i++) {
      if (UWts(user, i) > UWts(user, maxWtInd)) {
        maxWtInd = i;
      }
      if (UWts(user, i) != 0 ) { nnz++; }
    }
    avgUNNZ += nnz;

    int isHit = 0;
    //if (maxWtInd == exSetInd) {
    if (idx[0] == exSetInd) { // || idx[1] == exSetInd || idx[2] == exSetInd) {
       hits += 1;
       isHit = 1;
    }
    count += 1;

    float estExSetRMSE = estUExSetRMSE(userSets, maxWtInd);
    avgEstExSetRMSE += estExSetRMSE;
    
    diff += fabs(exSetRMSE - estExSetRMSE);
    //opFile << user << " " << exSetInd << " " << maxWtInd << " " 
    //  << exSetRMSE << " " << estExSetRMSE << " " << userSets.itemSets.size()
    //  << " " << nnz  << std::endl;
  }
  //opFile.close();
  std::cout << "Fraction of user hits: " << hits/count << std::endl;
  std::cout << "Avg diff b/w orig & est exSet: " << diff/count << std::endl;
  std::cout << "avgExSetRMSE: " << avgExSetRMSE/count 
    << " avgEstExSetRMSE: " << avgEstExSetRMSE/count << std::endl;
  std::cout << "avgNNZCoeff: " << avgUNNZ/count << std::endl;
}
