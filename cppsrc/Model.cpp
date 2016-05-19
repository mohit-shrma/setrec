#include "Model.h"


Model::Model(const Params &params) {
  
  nUsers      = params.nUsers;
  nItems      = params.nItems;
  facDim      = params.facDim;
  uReg        = params.uReg;
  uBiasReg    = params.uBiasReg;
  iBiasReg    = params.iBiasReg;
  uSetBiasReg = params.u_mReg;
  iReg        = params.iReg;
  learnRate   = params.learnRate;

  //random engine
  std::mt19937 mt(params.seed);
  std::uniform_real_distribution<> dis(0, 1);

  //initialize User factors and biases
  U = Eigen::MatrixXf(nUsers, facDim);
  uBias = Eigen::VectorXf(nUsers);
  uSetBias = Eigen::VectorXf(nUsers);

  for (int u = 0; u < nUsers; u++) {
    uBias(u) = dis(mt);
    uSetBias(u) = dis(mt);
    for (int k = 0; k < facDim; k++) {
      U(u, k) = dis(mt);
    }
  }

  //initialize item factors and biases
  V = Eigen::MatrixXf(nItems, facDim);
  iBias = Eigen::VectorXf(nItems);
  for (int item = 0; item < nItems; item++) {
    iBias(item) = dis(mt);
    for (int k = 0; k < facDim; k++) {
      V(item, k) = dis(mt);
    }
  }
  
  //init global bias
  gBias = 0;
}


Model::Model(const Params &params, const char* uFacName, 
    const char* iFacName):Model(params) {
  readEigenMat(uFacName, U, nUsers, facDim);
  readEigenMat(iFacName, V, nItems, facDim);
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
      auto items = uSet.itemSets[i].first;
      setScore = estSetRating(user, items);
      diff = setScore - uSet.itemSets[i].second;
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


float Model::objective(const std::vector<UserSets>& uSets, gk_csr_t *mat) {
  
  float obj = 0.0, uRegErr = 0.0, iRegErr = 0.0;
  float norm, setScore, diff;
  int user, nSets = 0;
  
  for (auto&& uSet: uSets) {
    user = uSet.user;
    for (size_t i = 0; i < uSet.itemSets.size(); i++) {
      auto items = uSet.itemSets[i].first;
      setScore = estSetRating(user, items);
      diff = setScore - uSet.itemSets[i].second;
      obj += diff*diff;
      nSets++;
    }
    for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
      int item = mat->rowind[ii];
      float r_ui = mat->rowval[ii];
      float r_ui_est = estItemRating(user, item);
      diff = r_ui_est - r_ui;
      obj += diff*diff;
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
      auto items = uSet.itemSets[i].first;
      float predSetScore = estSetRating(user, items);
      float diff = predSetScore - uSet.itemSets[i].second;
      rmse += diff*diff;
      nSets++;
    }
  }
  
  rmse = sqrt(rmse/nSets);
  return rmse;
}


float Model::rmse(gk_csr_t *mat) {
  float rmse = 0;
  float r_ui, r_ui_est, diff;
  int nnz = 0;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      r_ui = mat->rowval[ii];
      r_ui_est = estItemRating(u, item);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }
  rmse = sqrt(rmse/nnz);
  return rmse;
}


//compute RMSE for items in the sets
float Model::rmse(const std::vector<UserSets>& uSets, gk_csr_t *mat) {
  float rmse = 0, r_ui_est, r_ui, diff;
  int nnz = 0, item, u;
  for (auto&& uSet: uSets) {
    u = uSet.user;
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (uSet.items.find(item) != uSet.items.end()) {
        //item present in set
        r_ui_est = estItemRating(u, item);
        r_ui = mat->rowval[ii];
        diff = r_ui - r_ui_est;
        rmse += diff*diff;
        nnz++;
      }
    }
  }
  rmse = sqrt(rmse/nnz);
  return rmse;
}


std::map<int, float> Model::itemRMSE(const std::vector<UserSets>& uSets,
    gk_csr_t *mat) {

  float rmse = 0, r_ui_est, r_ui, diff;
  int nnz = 0, item, u;
  std::map<int, float> itemSE, itemCount;

  for (auto const & uSet: uSets) {
    u = uSet.user;
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (uSet.items.find(item) != uSet.items.end()) {
        //item present in set
        r_ui_est = estItemRating(u, item);
        r_ui = mat->rowval[ii];
        diff = r_ui - r_ui_est;
        
        if (itemSE.find(item) == itemSE.end()) {
          itemSE[item] = 0;
          itemCount[item] = 0;
        }
        
        itemSE[item] += diff*diff;
        itemCount[item] += 1;

        rmse += diff*diff;
        nnz++;
      }
    }
  }

  for (auto const & kv: itemSE) {
    int item = kv.first;
    itemSE[item] = sqrt(itemSE[item]/itemCount[item]);
  }

  rmse = sqrt(rmse/nnz);
  return itemSE;
}


float Model::spearmanRankN(gk_csr_t *mat, int N) {
  int item, nUsers = 0;
  std::vector<float> actualRatings, predRatings;
  float uSpearman, avgSpearMan = 0;
  for (int u = 0; u < mat->nrows; u++) {
    actualRatings.clear();
    predRatings.clear();
    for (int ii = mat->rowptr[u], j = 0; 
        ii < mat->rowptr[u+1] && j < N; ii++, j++) { 
      item = mat->rowind[ii];
      actualRatings.push_back(mat->rowval[ii]);
      predRatings.push_back(estItemRating(u, item));
    }
    uSpearman = spearmanRankCorrN(actualRatings, predRatings, N);
    if (uSpearman != uSpearman) {
      //NaN check
      continue;
    }
    avgSpearMan += uSpearman;
    nUsers++;
  }
  avgSpearMan = avgSpearMan/nUsers;
  return avgSpearMan;
}


float Model::spearmanRankN(gk_csr_t *mat, const std::vector<UserSets>& uSets, 
    int N) {
  int item, nUsers = 0;
  std::vector<float> actualRatings, predRatings;
  float uSpearman, avgSpearMan = 0;

  for (auto&& uSet: uSets) {
    int u = uSet.user;
    auto setItems = uSet.items;
    actualRatings.clear();
    predRatings.clear();
    for (int ii = mat->rowptr[u], j = 0; 
        ii < mat->rowptr[u+1] && j < N; ii++) {
      item = mat->rowind[ii];
      if (setItems.find(item) == setItems.end()) {
        //item not in userSet
        actualRatings.push_back(mat->rowval[ii]);
        predRatings.push_back(estItemRating(u, item));
        j++;
      }
    }
    uSpearman = spearmanRankCorrN(actualRatings, predRatings, N);
    if (uSpearman != uSpearman) {
      //NaN check
      continue;
    }
    avgSpearMan += uSpearman;
    nUsers++;
  }
  avgSpearMan = avgSpearMan/nUsers;
  return avgSpearMan;
}


//compute iversions by ranking items not present in user's sets
float Model::inversionCount(gk_csr_t *mat, const std::vector<UserSets>& uSets, 
    int N) {
  int item, nUsers = 0;
  std::vector<std::pair<int, float>> actualItemRatings, predItemRatings;
  float uInvCount, avgInvCount = 0;

  for (auto&& uSet: uSets) {
    int u = uSet.user;
    auto setItems = uSet.items;
    actualItemRatings.clear();
    predItemRatings.clear();
    for (int ii = mat->rowptr[u], j = 0; 
        ii < mat->rowptr[u+1] && j < N; ii++) {
      item = mat->rowind[ii];
      if (setItems.find(item) == setItems.end()) {
        //item not in userSet
        actualItemRatings.push_back(std::make_pair(item, mat->rowval[ii]));
        predItemRatings.push_back(std::make_pair(item, estItemRating(u, item)));
        j++;
      }
    }
    
    std::sort(actualItemRatings.begin(), actualItemRatings.end(), descComp);
    std::sort(predItemRatings.begin(), predItemRatings.end(), descComp);

    uInvCount = inversionCountPairs(actualItemRatings, predItemRatings);

    avgInvCount += uInvCount;
    nUsers++;
  }
  avgInvCount = avgInvCount/nUsers;
  return avgInvCount;
}


float Model::invertRandPairCount(gk_csr_t *mat, 
    const std::vector<UserSets>& uSets, int N, int seed) {
  int item, nUsers = 0;
  std::vector<std::pair<int, float>> actualItemRatings, predItemRatings;
  float uInvCount, avgInvCount = 0;

  std::mt19937 mt(seed);

  for (auto&& uSet: uSets) {
    int u = uSet.user;
    auto setItems = uSet.items;
    actualItemRatings.clear();
    predItemRatings.clear();
    for (int ii = mat->rowptr[u], j = 0; 
        ii < mat->rowptr[u+1] && j < N; ii++) {
      item = mat->rowind[ii];
      if (setItems.find(item) == setItems.end()) {
        //item not in userSet
        actualItemRatings.push_back(std::make_pair(item, mat->rowval[ii]));
        predItemRatings.push_back(std::make_pair(item, estItemRating(u, item)));
        j++;
      }
    }
    
    std::sort(actualItemRatings.begin(), actualItemRatings.end(), descComp);
    std::sort(predItemRatings.begin(), predItemRatings.end(), descComp);

    //select 2 items at random from the list
    std::uniform_int_distribution<int> dist(0, actualItemRatings.size());
    int p = dist(mt);
    int q = dist(mt);
    while (actualItemRatings[p].second == actualItemRatings[q].second) {
      q = dist(mt);
    }
    pItem = actualItemRatings[p].first;
    qItem = actualItemRatings[q].first;
    
    auto predPInd = std::find_if(predItemRatings.begin(), 
        predItemRatings.end(), 
        [&pItem] (std::pair<int, float> itemRating) { 
          return itemRating.first == pItem;
        });
    auto predQInd = std::find_if(predItemRatings.begin(), 
        predItemRatings.end(), 
        [&qItem] (std::pair<int, float> itemRating) { 
          return itemRating.first == qItem;
        });
    
    uInvCount = 0;
    if (!((p < q && predPInd < predQInd) 
          || (p > q && predPInd > predQInd))) {
       uInvCount++;
    }

    avgInvCount += uInvCount;
    nUsers++;
  }
  avgInvCount = avgInvCount/nUsers;
  return avgInvCount;
}


float Model::recallTopN(gk_csr_t *mat, const std::vector<UserSets>& uSets,
    int N) {
  float recN = 0;
  int uCount = 0;
  std::vector<std::pair<int, float>> itemPredRatings;
  std::vector<std::pair<int, float>> itemActRatings;
  std::unordered_set<int> predTopN;
  for (auto&& uSet: uSets) {
    int u = uSet.user;
    auto setItems = uSet.items;
    itemPredRatings.clear();
    itemActRatings.clear();
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      if (setItems.find(item) == setItems.end()) {
        //item not in user set
        itemPredRatings.push_back(std::make_pair(item, 
              estItemRating(u, item)));
        itemActRatings.push_back(std::make_pair(item, mat->rowval[ii]));
      }
    }
    
    if (itemPredRatings.size() == 0) {
      continue;
    }

    //arrange such that Nth element is in its place
    std::nth_element(itemActRatings.begin(), itemActRatings.begin()+(N-1), 
        itemActRatings.end(), descComp);
    std::nth_element(itemPredRatings.begin(), itemPredRatings.begin()+(N-1), 
        itemPredRatings.end(), descComp);
    
    predTopN.clear();
    for (int j = 0; j < N && j < (int)itemPredRatings.size(); j++) {
      predTopN.insert(itemPredRatings[j].first);
    }
    
    int overlapCt = 0;
    for (int j = 0; j < N && j < (int)itemPredRatings.size(); j++) {
      auto itemRating = itemActRatings[j];
      if (predTopN.find(itemRating.first) != predTopN.end()) {
        //found in predicted top N
        overlapCt++;
      }
    }
    recN += (float)overlapCt/predTopN.size();
    uCount++;
  }
  
  recN = recN/uCount;
  return recN;
}


float Model::recallTopN(gk_csr_t *mat, const std::vector<UserSets>& uSets,
    std::unordered_set<int>& invalUsers, int N) {
  float recN = 0;
  int uCount = 0;
  std::vector<std::pair<int, float>> itemPredRatings;
  std::vector<std::pair<int, float>> itemActRatings;
  std::unordered_set<int> predTopN;
  for (auto&& uSet: uSets) {
    if (invalUsers.find(uSet.user) != invalUsers.end()) {
      //found invalid user
      continue;
    }

    int u = uSet.user;
    auto setItems = uSet.items;
    itemPredRatings.clear();
    itemActRatings.clear();
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      if (setItems.find(item) == setItems.end()) {
        //item not in user set
        itemPredRatings.push_back(std::make_pair(item, 
              estItemRating(u, item)));
        itemActRatings.push_back(std::make_pair(item, mat->rowval[ii]));
      }
    }
    
    if (itemPredRatings.size() == 0) {
      continue;
    }

    //arrange such that Nth element is in its place
    std::nth_element(itemActRatings.begin(), itemActRatings.begin()+(N-1), 
        itemActRatings.end(), descComp);
    std::nth_element(itemPredRatings.begin(), itemPredRatings.begin()+(N-1), 
        itemPredRatings.end(), descComp);
    
    predTopN.clear();
    for (int j = 0; j < N && j < (int)itemPredRatings.size(); j++) {
      predTopN.insert(itemPredRatings[j].first);
    }
    
    int overlapCt = 0;
    for (int j = 0; j < N && j < (int)itemPredRatings.size(); j++) {
      auto itemRating = itemActRatings[j];
      if (predTopN.find(itemRating.first) != predTopN.end()) {
        //found in predicted top N
        overlapCt++;
      }
    }
    recN += (float)overlapCt/predTopN.size();
    uCount++;
  }
  
  recN = recN/uCount;
  return recN;
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
    for (int u = 0; u < nUsers; u++) {
      for (int k = 0; k < facDim; k++) {
        uOpFile << U(u, k) << " ";
      }
      uOpFile << std::endl;
    }
    uOpFile.close();
  }

  //save V
  fName = opPrefix + "_" + sign + "_V.eigen";
  std::ofstream vOpFile(fName);
  if (vOpFile.is_open()) {
    for (int item = 0; item < nItems; item++) {
      for (int k = 0; k < facDim; k++) {
        vOpFile << V(item, k) << " ";
      }
      vOpFile << std::endl;
    }
    vOpFile.close();
  }

  //save user biases
  fName = opPrefix + "_" + sign + "_ubias";
  std::ofstream uBiasOpFile(fName);
  if (uBiasOpFile.is_open()) {
    for (int u = 0; u < nUsers; u++) {
      uBiasOpFile << uBias[u] << std::endl;
    }
    uBiasOpFile.close();
  }

  //save user set biases
  fName = opPrefix + "_" + sign + "_uSetBias";
  std::ofstream uSetBiasOpFile(fName);
  if (uSetBiasOpFile.is_open()) {
    for (int u = 0; u < nUsers; u++) {
      uSetBiasOpFile << uSetBias[u] << std::endl;
    }
    uSetBiasOpFile.close();
  }
  

  //save item biases
  fName = opPrefix + "_" + sign + "_ibias";
  std::ofstream iBiasOpFile(fName);
  if (iBiasOpFile.is_open()) {
    for (int item = 0; item < nItems; item++) {
      iBiasOpFile << iBias[item] << std::endl;
    }
    iBiasOpFile.close();
  }
  
  //save global bias
  fName = opPrefix + "_" + sign + "_gbias";
  std::ofstream gBiasOpfile(fName);
  if (gBiasOpfile.is_open()) {
    gBiasOpfile << gBias << std::endl;
    gBiasOpfile.close();
  }

}


void Model::load(std::string opPrefix) {
  std::string sign = modelSign();
  
  //load U
  std::string fName = opPrefix + "_" + sign + "_U.eigen";
  readEigenMat(fName.c_str(), U, nUsers, facDim);

  //load V
  fName = opPrefix + "_" + sign + "_V.eigen";
  readEigenMat(fName.c_str(), V, nItems, facDim);

  //load user biases
  fName = opPrefix + "_" + sign + "_ubias";
  std::vector<float> fVec = readFVector(fName.c_str());
  for (int u = 0; u < nUsers; u++) {
    uBias(u) = fVec[u];
  }

  //load user biases
  fName = opPrefix + "_" + sign + "_uSetBias";
  fVec = readFVector(fName.c_str());
  for (int u = 0; u < nUsers; u++) {
    uSetBias(u) = fVec[u];
  }

  //load item biases
  fName = opPrefix + "_" + sign + "_ibias";
  fVec = readFVector(fName.c_str());
  for (int item = 0; item < nItems; item++) {
    iBias(item) = fVec[item];
  }

  //load global bias
  fName = opPrefix + "_" + sign + "_gbias";
  std::ifstream ipFile(fName);
  if (ipFile.is_open()) {
    std::string line;
    if (getline(ipFile, line)) {
      gBias = std::stof(line);
    }
    ipFile.close();
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


bool Model::isTerminateModelWPart(Model& bestModel, const Data& data, int iter,
    int& bestIter, float& bestObj, float& prevObj, float& bestValRMSE,
    float& prevValRMSE) {

  bool ret = false;  
  float currObj = objective(data.trainSets, data.partTrainMat);
  float currValRMSE = -1;
  
  currValRMSE = rmse(data.valSets); 

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


bool Model::isTerminateModelWPart(Model& bestModel, const Data& data, int iter,
    int& bestIter, float& bestObj, float& prevObj) {

  bool ret = false;  
  float currObj = objective(data.trainSets, data.partTrainMat);
  

  if (iter > 0) {
    if (currObj < bestObj) {
      bestModel = *this;
      bestIter = iter;
      bestObj = currObj;
    } 
  
    if (iter - bestIter >= CHANCE_ITER) {
      //cant improve validation RMSE
      std::cout << "NOT CONVERGED VAL: bestIter:" << bestIter << " bestObj:" 
        << bestObj <<  " currIter:" << iter << " currObj: " << currObj << std::endl;
      ret = true;
    }
    
    
    if (fabs(prevObj - currObj) < EPS) {
      //objective converged
      std::cout << "CONVERGED OBJ:" << iter << " currObj:" << currObj;
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
    bestIter = iter;
  }
  
  prevObj = currObj;

  return ret;
}


bool Model::isTerminateRecallModel(Model& bestModel, const Data& data, int iter,
    int& bestIter, float& bestRecall, float& prevRecall, float& bestValRecall,
    float& prevValRecall, std::unordered_set<int>& invalidUsers) {

  bool ret = false;  
  float currRecall = recallTopN(data.ratMat, data.trainSets, invalidUsers, 10);
  float currValRecall = recallTopN(data.ratMat, data.valSets, invalidUsers, 10);
  
  if (iter > 0) {
    if (currValRecall > bestValRecall) {
      bestModel = *this;
      bestValRecall = currValRecall;
      bestIter = iter;
      bestRecall = currRecall;
    } 
  
    if (iter - bestIter >= 1000) {
      //cant improve validation RMSE
      std::cout << "NOT CONVERGED VAL: bestIter:" << bestIter << " bestRecall:" 
        << bestRecall << " bestValRecall: " << bestValRecall << " currIter:"
        << iter << " currRecall: " << currRecall << " currValRecall:" 
        << currValRecall << std::endl;
      ret = true;
    }
     
    /*
    if (fabs(prevRecall - currRecall) < EPS) {
      //objective converged
      std::cout << "CONVERGED Recall:" << iter << " currRecall:" << currRecall 
        << " bestValRecall:" << bestValRecall << std::endl;
      ret = true;
    }
    */ 
   
    /*
    if (fabs(prevValRecall - currValRecall) < EPS) {
      //Validation rmse converged
      std::cout << "CONVERGED VAL: bestIter:" << bestIter << " bestRecall:" 
        << bestRecall << " bestValRecall: " << bestValRecall << " currIter:"
        << iter << " currRecall: " << currRecall << " currValRecall:" 
        << currValRecall << std::endl;
      ret = true;
    }
    */
  }
  
  if (0 == iter) {
    bestRecall = currRecall;
    bestValRecall = currValRecall;
    bestIter = iter;
  }
  
  prevRecall = currRecall;
  prevValRecall = currValRecall;

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


