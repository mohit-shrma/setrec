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
  gamma       = params.rhoRMS;

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


//TODO: init train users and train items in models
float Model::estItemRating(int user, int item) {
  if (trainUsers.find(user) != trainUsers.end() && 
      trainItems.find(item) != trainItems.end()) {
    return (U.row(user)).dot(V.row(item));
  } 
  return 0;
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


float Model::objective(gk_csr_t *mat) {
  
  float obj = 0.0, uRegErr = 0.0, iRegErr = 0.0;
  float norm, diff, r_ui, r_ui_est;
  
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      r_ui = mat->rowval[ii];
      r_ui_est = estItemRating(u, item);
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


float Model::rmse(gk_csr_t *mat, std::unordered_set<int>& valItems) {
  float rmse = 0;
  float r_ui, r_ui_est, diff;
  int nnz = 0;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      if (valItems.find(item) == valItems.end()) {
        //not valid item
        continue;
      }
      r_ui = mat->rowval[ii];
      r_ui_est = estItemRating(u, item);
      diff = r_ui - r_ui_est;
      rmse += diff*diff;
      nnz++;
    }
  }
  std::cout << "nnz: " << nnz << " rmse: " << rmse << std::endl;
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
    const std::vector<UserSets>& uSets, int seed) {
  int item, nUsers = 0;
  std::unordered_set<int> missedPs, missedQs;
  std::vector<std::pair<int, float>> actualItemRatings, predItemRatings;
  float uInvCount, avgInvCount = 0;
  int missedP = 0, missedQ = 0;
  std::mt19937 mt(seed);

  for (auto&& uSet: uSets) {
    int u = uSet.user;
    
    if (invalidUsers.find(u) != invalidUsers.end()) {
      //skip if invalid user
      continue;
    }

    auto setItems = uSet.items;
    actualItemRatings.clear();
    predItemRatings.clear();
    for (int ii = mat->rowptr[u];ii < mat->rowptr[u+1]; ii++) {
      item = mat->rowind[ii];
      if (setItems.find(item) == setItems.end()) {
        //item not in userSet
        actualItemRatings.push_back(std::make_pair(item, mat->rowval[ii]));
        predItemRatings.push_back(std::make_pair(item, estItemRating(u, item)));
      }
    }
   
    if (actualItemRatings.size() <= 1) {
      continue;
    }

    std::sort(actualItemRatings.begin(), actualItemRatings.end(), descComp);
    std::sort(predItemRatings.begin(), predItemRatings.end(), descComp);

    //select 2 items at random from the list
    std::uniform_int_distribution<int> dist(0, actualItemRatings.size()-1);
    int p = dist(mt);
    int q = dist(mt);
    int nTries = 0;
    while (actualItemRatings[p].second == actualItemRatings[q].second 
        && nTries < 100) {
      q = dist(mt);
      nTries++;
    }

    if (actualItemRatings[p].second == actualItemRatings[q].second) {
      continue;
    }

    int pItem = actualItemRatings[p].first;
    int qItem = actualItemRatings[q].first;

    //check if pItem or qItem in training
    if (trainItems.find(pItem) == trainItems.end()) {
      missedP++;
      missedPs.insert(pItem);
    }
    
    if (trainItems.find(qItem) == trainItems.end()) {
      missedQ++;
      missedQs.insert(qItem);
    }

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
  std::cout << "users: " << trainUsers.size() << " items: " << trainItems.size() << std::endl;
  std::cout << "nUsers: " << nUsers << " avgInvCount: " << avgInvCount 
    << " missedP: " << missedP << " missedQ: " << missedQ
    << std::endl;
  //writeContainer(missedPs.begin(), missedPs.end(),  "missedPs.txt");
  //writeContainer(missedQs.begin(), missedQs.end(),  "missedQs.txt");
  //writeContainer(trainItems.begin(), trainItems.end(), "trainItems.txt");
  avgInvCount = avgInvCount/nUsers;
  return avgInvCount;
}


float Model::invertRandPairCount(
    std::vector<std::tuple<int, int, int>> allTriplets) {

  int correctCt = 0, incorrectCt = 0;
  
  for (auto&& triplet: allTriplets) {
    int user  = std::get<0>(triplet);
    int item1 = std::get<1>(triplet);
    int item2 = std::get<2>(triplet);
    
    if (invalidUsers.find(user) != invalidUsers.end()) {
      //skip if invalid user
      continue;
    }
    
    float rating1 = estItemRating(user, item1);
    float rating2 = estItemRating(user, item2);

    if (rating1 > rating2) {
      //correct
      correctCt++;
    } else {
      //incorrect
      incorrectCt++;
    }
  }
  std::cout << "Ord count: "  << correctCt+incorrectCt << " " << correctCt 
    << " " << incorrectCt << std::endl; 
  return (float)correctCt/(correctCt + incorrectCt);
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
    
    if (invalidUsers.find(u) != invalidUsers.end()) {
      //skip if invalid user
      continue;
    }

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


float Model::ratingsNDCG(
    std::map<int, std::map<int, float>> uRatings) {
  
  float avgNDCG = 0, nUsers = 0;
  std::vector<std::pair<int, float>> predItemRatings;
  std::vector<std::pair<int, float>> origItemRatings;
  std::vector<float> orig, pred;
  std::map<int, float> itemRatingMap;

  for (auto&& uRating: uRatings) {
    int user = uRating.first;
    
    if (invalidUsers.find(user) != invalidUsers.end()) {
      continue;
    }

    predItemRatings.clear();
    origItemRatings.clear();
    orig.clear();
    pred.clear();
    itemRatingMap.clear();

    for (auto&& itemRating: uRating.second) {
      auto item = itemRating.first;
      origItemRatings.push_back(std::make_pair(item, itemRating.second));
      float rating = estItemRating(user, item);
      predItemRatings.push_back(std::make_pair(item, rating));
      itemRatingMap[item] = itemRating.second;  
    }

    std::sort(origItemRatings.begin(), origItemRatings.end(), descComp);
    std::sort(predItemRatings.begin(), predItemRatings.end(), descComp);
    
    for (auto&& itemRating: origItemRatings) {
      orig.push_back(itemRating.second);
    }

    for (auto&& itemRating: predItemRatings) {
      pred.push_back(itemRatingMap[itemRating.first]);
    }
    
    avgNDCG += ndcg(orig, pred);
    nUsers += 1;
  }
  
  avgNDCG = avgNDCG/nUsers;
  std::cout << "nUsers: " << nUsers << " avgNDCG: " << avgNDCG << std::endl;
  return avgNDCG;
}


float Model::ratingsNDCGRel(
    std::map<int, std::map<int, float>> uRatings) {

  float avgNDCG = 0, ndcg = 0, nUsers = 0;
  std::vector<std::pair<int, float>> predItemRatings;
  std::vector<std::pair<int, float>> origItemRatings;
  std::vector<float> orig, pred;
  std::map<int, float> itemRatingMap;

  for (auto&& uRating: uRatings) {
    int user = uRating.first;
    
    if (invalidUsers.find(user) != invalidUsers.end()) {
      continue;
    }

    predItemRatings.clear();
    origItemRatings.clear();
    orig.clear();
    pred.clear();
    itemRatingMap.clear();

    for (auto&& itemRating: uRating.second) {
      auto item = itemRating.first;
      origItemRatings.push_back(std::make_pair(item, itemRating.second));
      float rating = estItemRating(user, item);
      predItemRatings.push_back(std::make_pair(item, rating));
      itemRatingMap[item] = itemRating.second;  
    }

    std::sort(origItemRatings.begin(), origItemRatings.end(), descComp);
    std::sort(predItemRatings.begin(), predItemRatings.end(), descComp);
   
    std::unordered_set<float> uniqRat;
    for (auto&& itemRating: origItemRatings) {
      orig.push_back(itemRating.second);
      uniqRat.insert(itemRating.second);
    }


    for (auto&& itemRating: predItemRatings) {
      pred.push_back(itemRatingMap[itemRating.first]);
    }
    
    ndcg = ndcgRel(orig, pred);
    
    //std::cout << "User: " << user << " " << origItemRatings.size() << " " 
    //  << ndcg << " ";
    //for (auto&& rat: uniqRat) {
    //for (auto&& rat: pred) {
    //  std::cout << rat << " ";
    //}
    //std::cout << std::endl;

    avgNDCG += ndcg;
    nUsers += 1;
  }
  
  avgNDCG = avgNDCG/nUsers;
  return avgNDCG;
}


float Model::ratingsNDCGRelRand(
    std::map<int, std::map<int, float>> uRatings,
    std::mt19937& mt) {
  
  float avgNDCG = 0, ndcg = 0, nUsers = 0;
  std::vector<std::pair<int, float>> predItemRatings;
  std::vector<std::pair<int, float>> origItemRatings;
  std::vector<float> orig, pred;
  std::map<int, float> itemRatingMap;

  for (auto&& uRating: uRatings ) {
    int user = uRating.first;
    
    if (invalidUsers.find(user) != invalidUsers.end()) {
      continue;
    }

    predItemRatings.clear();
    origItemRatings.clear();
    orig.clear();
    pred.clear();
    itemRatingMap.clear();

    for (auto&& itemRating: uRating.second) {
      auto item = itemRating.first;
      origItemRatings.push_back(std::make_pair(item, itemRating.second));
      float rating = estItemRating(user, item);
      predItemRatings.push_back(std::make_pair(item, rating));
      itemRatingMap[item] = itemRating.second;  
    }

    std::sort(origItemRatings.begin(), origItemRatings.end(), descComp);
    std::sort(predItemRatings.begin(), predItemRatings.end(), descComp);
   
    std::unordered_set<float> uniqRat;
    for (auto&& itemRating: origItemRatings) {
      orig.push_back(itemRating.second);
      uniqRat.insert(itemRating.second);
    }


    for (auto&& itemRating: predItemRatings) {
      pred.push_back(itemRatingMap[itemRating.first]);
    }
    
    std::shuffle(pred.begin(), pred.end(), mt);
    
    ndcg = ndcgRel(orig, pred);
    
    //std::cout << "User: " << user << " " << origItemRatings.size() << " " 
    //  << ndcg << " ";
    //for (auto&& rat: uniqRat) {
    //for (auto&& rat: pred) {
    //  std::cout << rat << " ";
    //}
    //std::cout << std::endl;

    avgNDCG += ndcg;
    nUsers += 1;
  }
  
  avgNDCG = avgNDCG/nUsers;
  return avgNDCG;
}


float Model::recallHit(const std::vector<UserSets>& uSets,
    std::map<int, int> uItems, 
    std::map<int, std::unordered_set<int>> ignoreUItems, int N) {
  
  std::vector<std::pair<int, float>> predRatings;
  
  float hits = 0, nUsers = 0;
  
  for(auto&& uSet: uSets) {
    int user = uSet.user;
    
    if (ignoreUItems.find(user) == ignoreUItems.end() || 
        uItems.find(user) == uItems.end() || 
        invalidUsers.find(user) != invalidUsers.end()) {
      continue;
    }

    auto setItems = uSet.items;
    
    predRatings.clear();
    for (auto&& item: trainItems) {
      //skip if item in user's set or in user's ignore set
      if (setItems.find(item) != setItems.end() || 
          ignoreUItems[user].find(item) != ignoreUItems[user].end()) {
        continue;
      }
      predRatings.push_back(std::make_pair(item, estItemRating(user, item)));
    }

    if (predRatings.size() == 0) {
      continue;
    }

    if (N > (int)predRatings.size()) {
      N = predRatings.size();
    }

    std::nth_element(predRatings.begin(), predRatings.begin()+(N - 1), 
        predRatings.end(), descComp);
    
    for (auto it = predRatings.begin(); it != predRatings.begin()+N; it++) {
      int item = (*it).first;
      if (item == uItems[user]) {
        hits += 1;
      }
    }
    nUsers += 1;
  }
  
  //std::cout << "hits: " << hits << " nUsers: " << nUsers << std::endl;

  return hits/nUsers;
}


std::pair<float, float> Model::ratingsNDCGPrecK(const std::vector<UserSets>& uSets,
    std::map<int, std::map<int, float>> uRatings,
    int N) {
  int nUsers = 0;
  float avgNDCG = 0, avgPrec = 0;
  std::vector<std::pair<int, float>> predRatings;
  std::vector<std::pair<int, float>> origRatings;

  for (auto && uSet: uSets) {
    int user = uSet.user;
    if (invalidUsers.find(user) != invalidUsers.end() ||
        uRatings.find(user) == uRatings.end()) {
      continue;
    }
    
    auto setItems = uSet.items;
    predRatings.clear();
    
    for (auto&& item: trainItems) {
      //skip if item in user's set or in user's ignore set
      if (setItems.find(item) != setItems.end()) {
        continue;
      }
      predRatings.push_back(std::make_pair(item, estItemRating(user, item)));
    }
     
    if (predRatings.size() == 0) {
      continue;
    }
    
    if (N > (int)predRatings.size()) {
      N = predRatings.size();
    }
    
    std::nth_element(predRatings.begin(), predRatings.begin()+(N - 1), 
        predRatings.end(), descComp);
    std::sort(predRatings.begin(), predRatings.begin()+N, descComp);

    float found = 0;
    std::vector<float> orig, pred;
    for (auto it = predRatings.begin(); it != predRatings.begin()+N; it++) {
      int item = (*it).first;
      float actRating  = 0;
      if (uRatings[user].count(item) != 0) {
        actRating = uRatings[user][item];
        found += 1;
      }
      pred.push_back(actRating);
      orig.push_back(actRating);  
    }

    std::sort(orig.begin(), orig.end(), std::greater<int>());

    avgNDCG += ndcg(orig, pred);
    avgPrec += found/N; 

    nUsers++;
  }
  
  avgNDCG = avgNDCG/nUsers;
  avgPrec = avgPrec/nUsers;
  
  //std::cout << "avgNDCG: " << avgNDCG << "avgPrec: " << avgPrec 
  //  << " nUsers: " << nUsers << std::endl; 

  return std::make_pair(avgNDCG, avgPrec);
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


std::pair<float, float> Model::precisionNCall(
    const std::vector<UserSets>& uSets, gk_csr_t *mat, 
    int N, float ratingThresh) {  
  float avgPrecN = 0;
  float oneCall = 0;
  int nUsers = 0;

  std::vector<std::pair<int, float>> actRatings;
  std::unordered_set<int> actItems;
  std::vector<std::pair<int, float>> predRatings;
  
  for (auto&& uSet: uSets) {
    int user = uSet.user;
    if (invalidUsers.find(user) != invalidUsers.end()) {
      //found invalid user
      continue;
    }

    auto setItems = uSet.items;
    
    //get actual ratings greater than the threshold
    actRatings.clear();
    actItems.clear();
    for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      if (rating < ratingThresh) {
        continue;
      }
      if (setItems.find(item) != setItems.end()) {
        //item not found in set
        continue;
      }
      actRatings.push_back(std::make_pair(item, rating));
      actItems.insert(item);
    }
    
    if (actItems.size() == 0) {
      continue;
    }

    //get predictions over train items except those in the set 
    predRatings.clear();
    for (auto&& item: trainItems) {
      if (setItems.find(item) != setItems.end()) {
        //item not found in set
        continue;
      }
      predRatings.push_back(std::make_pair(item, estItemRating(user, item))); 
    }
    
    //std::sort(actRatings.begin(), actRatings.end(), descComp);
    std::nth_element(predRatings.begin(), predRatings.begin() + (N - 1), 
        predRatings.end(), descComp);
    float uFound = 0;
    for (int i = 0; i < N; i++) {
      int predItem = predRatings[i].first;
      if (actItems.find(predItem) != actItems.end()) {
        //relevant item found
        uFound += 1;
       }
     }
    
    if ((int)actItems.size() < N) {
      avgPrecN += uFound/actItems.size();
    } else {
      avgPrecN += uFound/N;
    }
      
    if (uFound > 0) {
      oneCall += 1;
    }

    nUsers++;
  }
  
  oneCall = oneCall/nUsers;
  avgPrecN = avgPrecN/nUsers;
  std::cout << "nUsers: " << nUsers << " avgPrecN: " << avgPrecN 
    << " oneCall: " << oneCall << std::endl;
  return std::make_pair(avgPrecN, oneCall);
}


float Model::precisionN(gk_csr_t* testMat, gk_csr_t* valMat, gk_csr_t* trainMat,
    int N) {
  
  float avgPrecN = 0;
  int nUsers = 0;

  std::vector<std::pair<int, float>> actRatings, predRatings;
  std::unordered_set<int> actItems;
  std::unordered_set<int> uTrainItems;

  for (const auto& u : trainUsers) {

    actItems.clear();
    actRatings.clear();
    predRatings.clear();
    uTrainItems.clear();

    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      int item = testMat->rowind[ii];
      if (trainItems.find(item) == trainItems.end()) {
        continue;
      }
      float rating = testMat->rowval[ii];
      if (rating >= 4) {
        actItems.insert(item);
      }
    }
    
    for (int ii = valMat->rowptr[u]; ii < valMat->rowptr[u+1]; ii++) {
      int item = valMat->rowind[ii];
      if (trainItems.find(item) == trainItems.end()) {
        continue;
      }
      float rating = valMat->rowval[ii];
      if (rating >= 4) {
        actItems.insert(item);
      }
    }

    if (actItems.size() == 0) {
      continue;
    }
    
    for (int ii = trainMat->rowptr[u]; ii < trainMat->rowptr[u+1]; ii++) {
      int item = trainMat->rowind[ii];
      uTrainItems.insert(item);
    }
    
    for (auto&& item: trainItems) {
      //skip if present in training for user
      if (uTrainItems.find(item) != uTrainItems.end()) {
        continue;
      } 
      predRatings.push_back(std::make_pair(item, estItemRating(u, item)));
    }
    
    std::nth_element(predRatings.begin(), predRatings.begin() + (N - 1), 
        predRatings.end(), descComp);

    float uFound = 0;
    for (int i = 0; i < N; i++) {
      int predItem = predRatings[i].first;
      if (actItems.find(predItem) != actItems.end()) {
        //relevant item found
        uFound += 1;
       }
     }

    if ((int)actItems.size() < N) {
      avgPrecN += uFound/actItems.size();
    } else {
      avgPrecN += uFound/N;
    }

    nUsers++;
  }

  std::cout << "avgPrecN: " << avgPrecN << " nUsers: " << nUsers << std::endl;

  avgPrecN = avgPrecN/nUsers;
    
  return avgPrecN;
}


float Model::corrOrderedItems(
    std::vector<std::vector<std::pair<int, float>>> testRatings) {
  int nUsers = 0, nURatings = 0;
  float corrOrderedPairs = 0, nPairs = 0;
  int firstItem, secondItem;
  float firstRating, secondRating;
  float firstPredRating, secondPredRating;

  for (auto&& u: trainUsers) {
    nURatings = testRatings[u].size();
    
    if (0 == nURatings) {
      continue;
    }

    for (int i = 0; i < nURatings; i++) {
      firstItem = testRatings[u][i].first;
      if (trainItems.find(firstItem) == trainItems.end()) {
        continue;
      }
      firstRating = testRatings[u][i].second;
      firstPredRating = estItemRating(u, firstItem);
      for (int j = i+1; j < nURatings; j++) {
        secondItem = testRatings[u][j].first;
        if (trainItems.find(secondItem) == trainItems.end()) {
          continue;
        }
        secondRating = testRatings[u][j].second;
        if (firstRating == secondRating) {
          continue;
        }
        secondPredRating = estItemRating(u, secondItem);
        if (firstRating < secondRating && firstPredRating < secondPredRating) {
          corrOrderedPairs += 1; 
        } else if (firstRating > secondRating && firstPredRating > secondPredRating) {
          corrOrderedPairs += 1;
        }
        nPairs += 1;
      }  
    }
    nUsers++;
  }
  return corrOrderedPairs/nPairs;
}


float Model::fracCorrOrderedSets(const std::vector<UserSets>& uSets) {
  
  int nPairs = 0;
  float nCorrOrderedPairs = 0;

  for (auto&& uSet: uSets) {
    int nUSets = uSet.itemSets.size();
    int user = uSet.user;
    
    if (invalidUsers.find(user) != invalidUsers.end()) {
      continue;
    }

    for (int i = 0; i < nUSets; i++) {
      
      float r_ui     = uSet.itemSets[i].second;
      auto items     = uSet.itemSets[i].first;
      float r_ui_est = estSetRating(user, items);

      for (int j = i+1; j < nUSets; j++) {
        
        float r_uj     = uSet.itemSets[j].second;
        auto items     = uSet.itemSets[j].first;
        float r_uj_est = estSetRating(user, items);
        
        if (r_uj != r_ui) {
          if ((r_uj > r_ui && r_uj_est > r_ui_est) ||
              (r_uj < r_ui && r_uj_est < r_ui_est)) {
            nCorrOrderedPairs += 1;
          }

          nPairs++;
        }
    
      } 
    }
  }

  //std::cout << "nPairs: " << nPairs << " nCorrOrderedPairs: " 
  //  << nCorrOrderedPairs << std::endl;

  return nCorrOrderedPairs/nPairs;
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

  //save train users
  fName = opPrefix + "_" + sign + "_trainUsers";
  writeContainer(trainUsers.begin(), trainUsers.end(), fName.c_str());

  //save train items
  fName = opPrefix + "_" + sign + "_trainItems";
  writeContainer(trainItems.begin(), trainItems.end(), fName.c_str());

  //save invalid users
  fName = opPrefix + "_" + sign + "_invalidUsers";
  writeContainer(invalidUsers.begin(), invalidUsers.end(), fName.c_str());
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
  
  //load train users
  fName = opPrefix + "_" + sign + "_trainUsers";
  auto iVec = readVector(fName.c_str());
  for (auto&& user: iVec) {
    trainUsers.insert(user);
  }

  //load train items
  fName = opPrefix + "_" + sign + "_trainItems";
  iVec = readVector(fName.c_str());
  for (auto&& item: iVec) {
    trainItems.insert(item);
  }

  //load invalid users
  fName = opPrefix + "_" + sign + "_invalidUsers";
  iVec = readVector(fName.c_str());
  for (auto&& user: iVec) {
    invalidUsers.insert(user);
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
      bestModel   = *this;
      bestValRMSE = currValRMSE;
      bestIter    = iter;
      bestObj     = currObj;
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
    
  } else if (0  == iter) {
    bestObj     = currObj;
    bestValRMSE = currValRMSE;
    bestIter    = iter;
    bestModel   = *this;
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
    float& prevValRecall) {

  bool ret = false;  
  float currRecall = recallTopN(data.ratMat, data.trainSets, invalidUsers, 10);
  //float currValRecall = recallHit(data.trainSets, data.valUItems, 
  //    data.ignoreUItems, 10);
  float currValRecall = ratingsNDCGRel(data.valURatings);
  
  if (iter > 0) {
    if (currValRecall > bestValRecall) {
      bestModel     = *this;
      bestValRecall = currValRecall;
      bestIter      = iter;
      bestRecall    = currRecall;
    } 
  
    if (iter - bestIter >= CHANCE_ITER) {
      //cant improve validation RMSE
      std::cout << "NOT CONVERGED VAL: bestIter:" << bestIter << " bestRecall: " 
        << bestRecall << " bestValRecall: " << bestValRecall << " currIter: "
        << iter << " currRecall: " << currRecall << " currValRecall: " 
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
    bestRecall    = currRecall;
    bestValRecall = currValRecall;
    bestIter      = iter;
    bestModel     = *this;
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


bool Model::isTerminateRankSetModel(Model& bestModel, const Data& data, int iter, 
    int& bestIter, float& prevValRecall, float& bestValRecall) {
  bool ret = false;
  float currValRecall = fracCorrOrderedSets(data.testValMergeSets);
  
  if (iter > 0) {
    if (currValRecall > bestValRecall) {
      bestModel     = *this;
      bestValRecall = currValRecall;
      bestIter      = iter;
    } 
    
    if (iter - bestIter >= CHANCE_ITER) {
      //cant improve validation RMSE
      std::cout << "NOT CONVERGED VAL: bestIter:" << bestIter  
        << " bestValRecall: " << bestValRecall << " currIter: "
        << iter << " currValRecall: " << currValRecall << std::endl;
      ret = true;
    }
  
  }

  if (0 == iter) {
    bestValRecall = currValRecall;
    bestIter      = iter;
    bestModel     = *this;
  }

  prevValRecall = currValRecall;
  return ret;
}


