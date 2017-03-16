#include "ModelWtAverageAllRangeWBias.h"


float ModelWtAverageAllRangeWBias::estItemRating(int user, int item) {
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


void ModelWtAverageAllRangeWBias::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelWtAverageAllRangeWBias::train" << std::endl; 
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
        uBias(user) -= learnRate*((2.0*(r_us_est - r_us)*sumWt) + 2.0*uBiasReg*uBias(user));

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
          iBias(item) -= learnRate*(2.0*(r_us_est - r_us)*iBiasGrad
                                    + 2.0*iBiasReg*iBias(item));
        }
        
      }
    }

    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacBiasUsingRatMat(partUIRatingsTup);
    }

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

    //if (iter % 5 == 0) {
    if (true) {
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
          if (wtRMSE[minInd] > wtRMSE[i]) {
            minInd = i;
          } 
        }
        UWts.row(user).fill(0);
        UWts(user, minInd) = 1;
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
      if (iter%100 == 0 || iter == params.maxIter - 1) {
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
  float hits = 0, count = 0, diff = 0, avgExSetRMSE = 0, avgEstExSetRMSE = 0; 
  std::ofstream opFile("user_weights_esqp.txt");
  for (const auto& userSets: data.trainSets) {
    
    int user = userSets.user;
    auto exSetNRMSE = userSets.getTopExtremalSubsetWRMSE(data.ratMat);
    int exSetInd = exSetNRMSE.first;
    float exSetRMSE = exSetNRMSE.second;
    avgExSetRMSE += exSetRMSE;

    int maxWtInd = 0;
    for (int i = 1; i < 2*SET_SZ-1; i++) {
      if (UWts(user, i) > UWts(user, maxWtInd)) {
        maxWtInd = i;
      }
    }

    int isHit = 0;
    if (maxWtInd == exSetInd) {
       hits += 1;
       isHit = 1;
    }
    count += 1;

    float estExSetRMSE = estUExSetRMSE(userSets, maxWtInd);
    avgEstExSetRMSE += estExSetRMSE;
    
    diff += fabs(exSetRMSE - estExSetRMSE);
    opFile << user << " " << exSetInd << " " << maxWtInd << " " 
      << exSetRMSE << " " << estExSetRMSE << " " << userSets.itemSets.size()
      << std::endl;
  }
  opFile.close();
  std::cout << "Fraction of user hits: " << hits/count << std::endl;
  std::cout << "Avg diff b/w orig & est exSet: " << diff/count << std::endl;
  std::cout << "avgExSetRMSE: " << avgExSetRMSE/count 
    << " avgEstExSetRMSE: " << avgEstExSetRMSE/count << std::endl;
  */
}


void ModelWtAverageAllRangeWBias::trainQPSmooth(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelWtAverageAllRangeWBias::trainQPSmooth" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;

  std::vector<std::pair<int, float>> setItemRatings(SET_SZ);
  std::vector<Eigen::VectorXf> cumSumItemFactors(SET_SZ); 
  for (int i = 0; i < SET_SZ; i++) {
    cumSumItemFactors[i] = Eigen::VectorXf(facDim);
  }
  std::vector<float> cumSumPreds(SET_SZ);

  float biasGrad;
  Eigen::VectorXf cumFac(facDim);
  Eigen::VectorXf grad(facDim);
  Eigen::VectorXf tempGrad(facDim);
  Eigen::MatrixXf UGradSqAvg(nUsers, facDim);
  Eigen::MatrixXf VGradSqAvg(nItems, facDim);
  Eigen::VectorXf uBiasGradSqAvg(nUsers);
  Eigen::VectorXf iBiasGradSqAvg(nItems);
  
  UGradSqAvg.fill(0); VGradSqAvg.fill(0);
  uBiasGradSqAvg.fill(0); iBiasGradSqAvg.fill(0);

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

  auto uSetInds = getUserSetInds(data.trainSets);

  if (params.isMixRat) {
    std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
    updateFacBiasUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg, 
        uBiasGradSqAvg, iBiasGradSqAvg);
  }
  
  int nUserCh = 0;

  for (int iter = 0; iter < params.maxIter; iter++) {

    nUserCh = 0;
    std::shuffle(uSetInds.begin(), uSetInds.end(), mt);
    
    for (const auto& uSetInd: uSetInds) {
      int uInd = uSetInd.first;
      int setInd = uSetInd.second;
    
      const UserSets& uSet = data.trainSets[uInd];
      int user = uSet.user;
            
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
      biasGrad = (2.0*(r_us_est - r_us)*sumWt) + 2.0*uBiasReg*uBias(user);
      uBiasGradSqAvg(user) = 0.9*uBiasGradSqAvg(user) + 0.1*biasGrad*biasGrad;
      uBias(user) -= (learnRate/std::sqrt(uBiasGradSqAvg(user) + 1e-8))*biasGrad;

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
        biasGrad = 2.0*(r_us_est - r_us)*iBiasGrad + 2.0*iBiasReg*iBias(item);
        iBiasGradSqAvg(item) = 0.9*iBiasGradSqAvg(item) + 0.1*biasGrad*biasGrad;
        iBias(item) -= (learnRate/std::sqrt(iBiasGradSqAvg(item) + 1e-8))*biasGrad; 
      }
        
    }

    if (params.isMixRat) {
      std::shuffle(partUIRatingsTup.begin(), partUIRatingsTup.end(), mt);
      updateFacBiasUsingRatMatRMSProp(partUIRatingsTup, UGradSqAvg, VGradSqAvg, 
          uBiasGradSqAvg, iBiasGradSqAvg);
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
      if (iter % 250 == 0 || iter == params.maxIter - 1) {
        std::cout << "Iter:" << iter << " obj:" << prevObj  
          << " best Obj: " << bestObj << " val: " << prevValRMSE 
          << " best val: " << bestValRMSE 
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
  //std::ofstream opFile("user_weights_esqp.txt");
  for (const auto& userSets: data.trainSets) {
    
    int user = userSets.user;
    auto exSetNRMSE = userSets.getTopExtremalSubsetWRMSE(data.ratMat);
    int exSetInd = exSetNRMSE.first;
    float exSetRMSE = exSetNRMSE.second;
    avgExSetRMSE += exSetRMSE;

    int maxWtInd = 0;
    for (int i = 1; i < 2*SET_SZ-1; i++) {
      if (UWts(user, i) > UWts(user, maxWtInd)) {
        maxWtInd = i;
      }
    }
    
    float estExSetRMSE = estUExSetRMSE(userSets, maxWtInd);
    avgEstExSetRMSE += estExSetRMSE;

    int isHit = 0;
    if (maxWtInd == exSetInd || fabs(estExSetRMSE - avgEstExSetRMSE) <= 0.001) {
       hits += 1;
       isHit = 1;
    }
    count += 1;

    
    diff += fabs(exSetRMSE - estExSetRMSE);
    //opFile << user << " " << exSetInd << " " << maxWtInd << " " 
    //  << exSetRMSE << " " << estExSetRMSE << " " << userSets.itemSets.size()
    //  << std::endl;
  }
  //opFile.close();
  std::cout << "Fraction of user hits: " << hits/count << std::endl;
  std::cout << "Avg diff b/w orig & est exSet: " << diff/count << std::endl;
  std::cout << "avgExSetRMSE: " << avgExSetRMSE/count 
    << " avgEstExSetRMSE: " << avgEstExSetRMSE/count << std::endl;
  
}







