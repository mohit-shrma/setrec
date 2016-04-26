#include "Model.h"

Model::Model(const Params &params) {
  
  nUsers    = params.nUsers;
  nItems    = params.nItems;
  facDim    = params.facDim;
  uReg      = params.uReg;
  iReg      = params.iReg;
  learnRate = params.learnRate;

  //random engine
  std::mt19937 mt(params.seed);
  std::uniform_real_distribution<> dis(0, 1);

  //initialize User factors
  U = Eigen::MatrixXf(nUsers, facDim);
  for (int u = 0; u < nUsers; u++) {
    for (int k = 0; k < facDim; k++) {
      U(u, k) = dis(mt);
    }
  }

  //initialize item factors
  V = Eigen::MatrixXf(nItems, facDim);
  for (int item = 0; item < nItems; item++) {
    for (int k = 0; k < facDim; k++) {
      V(item, k) = dis(mt);
    }
  }

}


float Model::estItemRating(int user, int item) {
  return (U.row(user)).dot(V.row(item));
}


float Model::objective(const std::vector<UserSets>& uSets) {
  
  float obj = 0.0, uRegErr = 0.0, iRegErr = 0.0;
  float norm, setScore, diff;
  int user, nSets = 0;
  
  for (auto&& uSet: uSets) {
    user = uSet.user;
    for (size_t i = 0; i < uSet.itemSets.size(); i++) {
      auto items = uSet.itemSets[i];
      setScore = estSetRating(user, items);
      diff = setScore - uSet.setScores[i];
      obj += diff*diff;
      nSets++;
    }
  }
  
  norm = U.norm();
  uRegErr = uReg*norm*norm;

  norm = V.norm();
  iRegErr = iReg*norm*norm;

  obj += uRegErr + iRegErr;

  return obj;
}


float Model::rmse(const std::vector<UserSets>& uSets) {
  float rmse = 0;
  int nSets = 0;

  for (auto&& uSet: uSets) {
    int user = uSet.user;
    for (size_t i = 0; i < uSet.itemSets.size(); i++) {
      auto items = uSet.itemSets[i];
      float predSetScore = estSetRating(user, items);
      float diff = predSetScore - uSet.setScores[i];
      rmse += diff*diff;
      nSets++;
    }
  }
  
  rmse = sqrt(rmse/nSets);
  return rmse;
}


std::string Model::modelSign() {
  std::string sign;
  sign = std::to_string(facDim) + "_" + std::to_string(uReg) + "_" 
    + std::to_string(iReg) + "_" + std::to_string(learnRate);
  return sign;
}


void Model::save(std::string opPrefix) {
  std::string sign = modelSign();
  //save U
  std::string fName = opPrefix + "_" + sign + "_U.eigen";
  std::ofstream uOpFile(fName);
  if (uOpFile.is_open()) {
    uOpFile << U << std::endl;
    uOpFile.close();
  }

  //save V
  fName = opPrefix + "_" + sign + "_V.eigen";
  std::ofstream vOpFile(fName);
  if (vOpFile.is_open()) {
    vOpFile << V << std::endl;
    vOpFile.close();
  }
}


bool Model::isTerminateModel(Model& bestModel, const Data& data, int iter,
    int& bestIter, float& bestObj, float& prevObj, float& bestValRMSE,
    float& prevValRMSE) {

  bool ret = false;  
  float currObj = objective(data.trainSets);
  float currValRMSE = -1;
  
  currValRMSE = rmse(data.valSets); 

  if (iter > 0) {
    if (currValRMSE < bestValRMSE) {
      bestModel = *this;
      bestValRMSE = currValRMSE;
      bestIter = iter;
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

    if (fabs(prevValRMSE - currValRMSE) < EPS) {
      //Validation rmse converged
      std::cout << "CONVERGED VAL: bestIter:" << bestIter << " bestObj:" 
        << bestObj << " bestValRMSE: " << bestValRMSE << " currIter:"
        << iter << " currObj: " << currObj << " currValRMSE:" 
        << currValRMSE << std::endl;
      ret = true;
    }
    
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

bool Model::isTerminateModel(Model& bestModel, const Data& data, int iter,
    int& bestIter, float& bestObj, float& prevObj) {

  bool ret = false;  
  float currObj = objective(data.trainSets);
  

  if (iter > 0) {
    if (currObj < bestObj) {
      bestModel = *this;
      bestObj = currObj;
      bestIter = iter;
    } 
  
    if (iter - bestIter >= CHANCE_ITER) {
      //cant improve validation RMSE
      std::cout << "NOT CONVERGED obj: bestIter:" << bestIter << " bestObj:" 
        << bestObj << " bestValRMSE: " << " currIter:"
        << iter << " currObj: " << currObj  << std::endl;
      ret = true;
    }
    
    if (fabs(prevObj - currObj) < EPS) {
      //objective converged
      std::cout << "CONVERGED OBJ:" << iter << " currObj:" << currObj 
        << std::endl;
      ret = true;
    }
    
  }
  
  if (0 == iter) {
    bestObj = currObj;
    bestIter = iter;
  }
  
  prevObj = currObj;

  return ret;
}
