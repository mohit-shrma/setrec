#include "ModelWtAverage.h"

void ModelWtAverage::estSetRatings(int user, const std::vector<int>& items, 
    float& r_us1, float& r_us2, float& r_us3) {
  
  r_us1 = 0;
  r_us2 = 0;
  r_us3 = 0;
  int setSz = items.size();
  
  std::vector<float> preds(setSz, 0);
  //get predictions
  for (int i = 0; i < setSz; i++) {
    int item = items[i];
    preds[i] = estItemRating(user, item);
  }
  //sort predictions
  std::sort(preds.begin(), preds.end());

  for (int i = 0; i < setSz; i++) {
    if (i < setSz-1) {
      r_us1 += preds[i];
    }  
    
    r_us2 += preds[i];

    if (i > 0) {
      r_us3 += preds[i];
    }
  }
 
  r_us1 /= (setSz - 1);
  r_us2 /= setSz;
  r_us3 /= (setSz - 1);
}


float ModelWtAverage::estSetRating(int user, std::vector<int>& items) {
  
  float r_us = 0; 
  float r_us1 = 0, r_us2 = 0, r_us3 = 0;
  int setSz = items.size();
  float w1 = UWts(user, 0);
  float w2 = UWts(user, 1);
  float w3 = UWts(user, 2);

  //r_us += ((setSz-1)*UWts(user, 0) + setSz*UWts(user, 1) 
  //    + (setSz-1)*UWts(user, 2))*uBias(user);
  
  std::vector<float> preds(setSz, 0);
  //get predictions
  for (int i = 0; i < setSz; i++) {
    int item = items[i];
    preds[i] = estItemRating(user, item);
  }
  //sort predictions
  std::sort(preds.begin(), preds.end());

  for (int i = 0; i < setSz; i++) {
    if (i < setSz-1) {
      r_us1 += preds[i];
    }  
    
    r_us2 += preds[i];

    if (i > 0) {
      r_us3 += preds[i];
    }
  }
 
  r_us1 /= (setSz - 1);
  r_us2 /= setSz;
  r_us3 /= (setSz - 1);
  
  r_us = w1*r_us1 + w2*r_us2 + w3*r_us3;

  return r_us;
}


void ModelWtAverage::train(const Data& data, const Params& params, 
    Model& bestModel) {
  std::cout << "ModelWtAverage::train" << std::endl; 
  std::cout << "Objective: " << objective(data.trainSets) << std::endl;
  
  Eigen::VectorXf sumFirstItemFactors(facDim);
  Eigen::VectorXf sumAllItemFactors(facDim);
  Eigen::VectorXf sumLastItemFactors(facDim);

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
          std::cerr << "!! zero size itemset found !!" << std::endl; 
          continue;
        }
        
        int setSz = items.size();

        //sort predicted ratings in set in ascending order
        std::vector<std::pair<int, float>> itemRatings;
        for (auto item: items) {
          itemRatings.push_back(std::make_pair(item, 
                estItemRating(user, item)));
        }
        std::sort(itemRatings.begin(), itemRatings.end(), ascComp);
       
        sumFirstItemFactors.fill(0);
        sumAllItemFactors.fill(0);
        sumLastItemFactors.fill(0);
        float w1 = UWts(user, 0);
        float w2 = UWts(user, 1);
        float w3 = UWts(user, 2);
        float r_us_est = 0;
        float r_us1 = 0, r_us2 = 0, r_us3 = 0;
        for (int i = 0; i < setSz; i++) {
          int item = itemRatings[i].first;
          float rating = itemRatings[i].second;
          sumAllItemFactors += V.row(item);
          
          if (i < setSz-1) {
            r_us1 += rating;
          }  
          
          r_us2 += rating;

          if (i > 0) {
            r_us3 += rating;
          }

        }
        r_us1 /= (setSz - 1);
        r_us2 /= setSz;
        r_us3 /= (setSz - 1);
        r_us_est = w1*r_us1 + w2*r_us2 + w3*r_us3;

        sumFirstItemFactors = sumAllItemFactors - V.row(itemRatings[setSz-1].first);
        sumLastItemFactors = sumAllItemFactors - V.row(itemRatings[0].first);

        //user gradient
        grad = 2.0*(r_us_est - r_us)*(  (w1/(setSz-1))*sumFirstItemFactors 
                                      + (w2/setSz)*sumAllItemFactors 
                                      + (w3/(setSz-1))*sumLastItemFactors)
                                    + 2.0*uReg*U.row(user).transpose();

        //update user
        U.row(user) -= learnRate*(grad.transpose());
        
        //update user bias
        uBias(user) -= learnRate*((2.0*(r_us_est - r_us)*(w1 + w2 + w3)) + 2.0*uBiasReg*uBias(user));

        //update items
        float iBiasGrad;
        grad = 2.0*(r_us_est - r_us)*U.row(user);
        for (int i = 0; i < setSz; i++) {
          int item = itemRatings[i].first;
          tempGrad = 2.0*iReg*V.row(item).transpose(); 
          if (0 == i) {
            tempGrad  += grad*(w1/(setSz-1) + w2/setSz);
            iBiasGrad = w1/(setSz-1) + w2/setSz; 
          } else if (setSz-1 == i) {
            tempGrad  += grad*(w2/setSz + w3/(setSz-1));
            iBiasGrad = w2/setSz + w3/(setSz-1); 
          } else { 
            tempGrad  += grad*(w1/(setSz-1) + w2/setSz +  w3/(setSz-1));
            iBiasGrad = w1/(setSz-1) + w2/setSz +  w3/(setSz-1); 
          }
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
        float r_us1, r_us2, r_us3; 
        for (int i = 0; i < nSets; i++) {
          auto&& itemsSetNRating = uSet.itemSets[i];
          estSetRatings(user, itemsSetNRating.first, r_us1, r_us2, r_us3);
          Q(i, 0) = r_us1;
          Q(i, 1) = r_us2;
          Q(i, 2) = r_us3;
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
        alglib::real_1d_array bndl = "[0.0, 0.0, 0.0]";
        alglib::real_1d_array bndu = "[10.0, 10.0, 10.0]";
        
        //starting point
        alglib::real_1d_array x0;
        x0.setlength(nWts);
        for (int i = 0; i < nWts; i++) {
          x0[i] = UWts(user, i);
        }
        
        //constraint: x0 + x1 + x2 = 1;
        alglib::real_2d_array constr = "[[1.0, 1.0, 1.0, 1]]";
        alglib::integer_1d_array ct = "[0]"; // = constraint 

        //solution
        alglib::real_1d_array x;

        alglib::minqpstate state;
        alglib::minqpreport rep;
      
        //scaling parameter indicating that model variables are in same scale
        alglib::real_1d_array s = "[1,1,1]";

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
