#include "util.h"


bool descComp(std::pair<int, float>& a, std::pair<int, float>& b) {
  return a.second > b.second;
}


bool ascComp(std::pair<int, float>& a, std::pair<int, float>& b) {
  return a.second < b.second;
}


void removeInvalUIFrmSets(std::vector<UserSets>& uSets, 
    std::unordered_set<int>& valUsers, std::unordered_set<int>& valItems) {

  auto it = std::begin(uSets);
  while (it != std::end(uSets)) {
    //remove if user not valid
    int user = (*it).user;
    if (valUsers.find(user) == valUsers.end()) {
      //remove invalid user
      it = uSets.erase(it);
      continue;
    }

    //remove sets w invalid item
    (*it).removeInvalSets(valItems);

    //remove if no set left
    if ((*it).itemSets.size() == 0) {
      it = uSets.erase(it);
      continue;
    }
    
    ++it;
  }

}


void userItemsFrmSets(std::vector<UserSets>& uSets, 
    std::unordered_set<int>& users, std::unordered_set<int>& items) {
  users.clear();
  items.clear();
  for (auto&& uSet: uSets) {
    users.insert(uSet.user);
    for (auto&& itemSet: uSet.itemSets) {
      for (auto&& item: itemSet.first) {
        items.insert(item);
      }
    }
  }
}


std::map<int, int> getItemFreq(std::vector<UserSets>& uSets) {
  std::map<int, int> itemFreq;
  for (auto&& uSet: uSets) {
    for (auto&& itemSet: uSet.itemSets) {
      for (auto&& item: itemSet.first) {
        if (itemFreq.find(item) == itemFreq.end()) {
          itemFreq[item] = 0;
        }
        itemFreq[item]++;
      }
    }
  }
  return itemFreq;
}


void removeOverUnderRatedSets(std::vector<UserSets>& uSets, gk_csr_t* ratMat) {
  //remove over-under rated sets
  auto it = std::begin(uSets);
  while (it != std::end(uSets)) {
    (*it).removeOverUnderRatedSets(ratMat);
    if ((*it).itemSets.size() == 0) {
      it = uSets.erase(it);
    } else {
      ++it;
    }  
  }
}


//remove sets which contain items not present in valid 
void removeSetsWOValItems(std::vector<UserSets>& uSets, 
    std::unordered_set<int>& valItems) {
  auto it = std::begin(uSets);
  while (it != std::end(uSets)) {
    (*it).removeInvalSets(valItems);
    if ((*it).itemSets.size() == 0) {
      it = uSets.erase(it);
    } else {
      ++it;
    }
  }
}


void removeSetsWOVal(std::vector<UserSets>& uSets,
    std::unordered_set<int>& valUsers, std::unordered_set<int>& valItems) {
  
  auto it = std::begin(uSets);
  
  while (it != std::end(uSets)) {
    bool isRemoveU = false;
    
    if (valUsers.find((*it).user) == valUsers.end()) {
      //user not found
      isRemoveU = true;
    } else {
      (*it).removeInvalSets(valItems);
      if ((*it).itemSets.size() == 0) {
        isRemoveU = true;
      }
    }

    if (isRemoveU) {
      it = uSets.erase(it);
    } else {
      ++it;
    }

  }

}


void removeSetsWInvalUsers(std::vector<UserSets>& uSets,
    std::unordered_set<int>& inValUsers) {
  
  auto it = std::begin(uSets);
  
  while (it != std::end(uSets)) {
    bool isRemoveU = false;
    
    if (inValUsers.find((*it).user) != inValUsers.end()) {
      //invalid user found
      isRemoveU = true;
    }

    if (isRemoveU) {
      it = uSets.erase(it);
    } else {
      ++it;
    }

  }

}


std::vector<std::map<int, float>> getUIRatings(gk_csr_t *mat) {

  int nUsers = mat->nrows;
  std::vector<std::map<int, float>> uiRatings(nUsers);
  
  for (int u = 0; u < nUsers; u++) {
    std::map<int, float> itemRatings;
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      itemRatings[item] = rating;
    }
    uiRatings[u] = itemRatings;
  }
  
  return uiRatings;
}


std::vector<std::vector<std::pair<int, float>>> getUIRatings(gk_csr_t* testMat, 
    gk_csr_t* valMat, int nUsers) {
  std::vector<std::vector<std::pair<int, float>>> uiRatings(nUsers);
  
  if (testMat->nrows != valMat->nrows || testMat->nrows != nUsers 
      || valMat->nrows != nUsers) {
    std::cerr << "Users mismatch" << std::endl;
    exit(0);
  }

  for (int u = 0; u < nUsers; u++) {
    for (int ii = testMat->rowptr[u]; ii < testMat->rowptr[u+1]; ii++) {
      uiRatings[u].push_back(std::make_pair(testMat->rowind[ii], 
            testMat->rowval[ii]));
    }
    for (int ii = valMat->rowptr[u]; ii < valMat->rowptr[u+1]; ii++) {
      uiRatings[u].push_back(std::make_pair(valMat->rowind[ii], 
            valMat->rowval[ii]));
    }
  }

  return uiRatings;
}



std::vector<std::tuple<int, int, float>> getUIRatingsTup(gk_csr_t* mat) {
  std::vector<std::tuple<int, int, float>> uiRatings;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      uiRatings.push_back(std::make_tuple(u, item, rating));
    }
  }
  return uiRatings;
}

//will return rating > lb
std::vector<std::tuple<int, int, float>> getUIRatingsTup(gk_csr_t* mat, 
    float lb) {
  std::vector<std::tuple<int, int, float>> uiRatings;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      float rating = mat->rowval[ii];
      if (rating <= lb) {
        continue;
      }
      uiRatings.push_back(std::make_tuple(u, item, rating));
    }
  }
  return uiRatings;
}


float inversionCountPairs(std::vector<std::pair<int, float>> actualItemRatings,
    std::vector<std::pair<int, float>> predItemRatings) {
  float invCount = 0;
  for (size_t i = 0; i < predItemRatings.size(); i++) {    
    auto item   = predItemRatings[i].first;
    auto rating = predItemRatings[i].second;
    auto itemInd = std::find_if(actualItemRatings.begin(), 
        actualItemRatings.end(), 
        [&item] (std::pair<int, float> itemRating) { 
          return itemRating.first == item;
        });
    for (size_t j = i+1; j < predItemRatings.size(); j++) {
      int qItem = predItemRatings[j].first;
      //search for query item in actual items
      auto qInd = std::find_if(actualItemRatings.begin(),
          actualItemRatings.end(),
          [&qItem] (std::pair<int, float> itemRating) {
            return itemRating.first == qItem;
          });
      if (qInd < itemInd && rating != (*qInd).second) {
        //inversion
        invCount += 1;
      }
    }
  }
  return invCount;
}


float meanRating(gk_csr_t *mat) {
  float mu = 0;
  int nnz = 0;
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      float r_ui = mat->rowval[ii];
      mu += r_ui;
      nnz++;
    }
  }
  return mu/nnz;
}


std::pair<std::unordered_set<int>, std::unordered_set<int>> getUserItems(
    gk_csr_t *mat) {
  std::unordered_set<int> users, items;
  for (int u = 0; u < mat->nrows; u++) {
    if (mat->rowptr[u+1] - mat->rowptr[u] > 0) {
      users.insert(u);
    }
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      items.insert(item);
    }
  }
  return std::make_pair(users, items);
}


std::pair<std::unordered_set<int>, std::unordered_set<int>> getUserItems(
    const std::vector<UserSets>& uSets) {
  std::unordered_set<int> users, items;
  
  for (auto&& uSet: uSets) {
    if (uSet.itemSets.size() > 0) {
      users.insert(uSet.user);
    }
    for (auto&& itemSet: uSet.itemSets) {
      for (auto&& item: itemSet.first) {
        items.insert(item);
      }
    }
  }

  return std::make_pair(users, items);
}


float dcgRel(std::vector<float> ord) {
  float dcg = 0;
  for (size_t i = 0; i < ord.size(); i++) {
    dcg += (std::pow(2, ord[i]) - 1) / (std::log2(1+(i+1)));
  }
  return dcg;
}


float ndcgRel(std::vector<float> orig, std::vector<float> pred) {
  float ndcg = 0;
  float predNDCG = dcgRel(pred);
  float origNDCG = dcgRel(orig);
  if (0 == predNDCG || 0 == origNDCG) {
    ndcg = 0;
  } else {
    ndcg = predNDCG/origNDCG;
  }
  return ndcg;
}


float dcg(std::vector<float> ord) {
  float dcg = 0;
  dcg += ord[0];
  for (size_t i = 1; i < ord.size(); i++) {
    dcg +=  ord[i] / std::log2(i+1);
  }
  return dcg;
}


float ndcg(std::vector<float> orig, std::vector<float> pred) {
  float ndcg = 0;
  float predNDCG = dcg(pred);
  float origNDCG = dcg(orig);
  if (0 == predNDCG || 0 == origNDCG) {
    ndcg = 0;
  } else {
    ndcg = predNDCG/origNDCG;
  }
  return ndcg;
}


//return {item1: [item2, item3]} such ru_item1 > item2 & item3
std::map<int, std::unordered_set<int>> getInvertItemPairs(
    std::vector<std::pair<int, float>> itemRatings, int maxTriplets,
    std::mt19937& mt) {
  
  std::map<int, std::unordered_set<int>> invertedPairs;
  int ind, secInd;
  const int nRatings = (int)itemRatings.size();
  if (maxTriplets > nRatings/2) {
    maxTriplets = nRatings/2;
  }

  std::sort(itemRatings.begin(), itemRatings.end(), descComp);
  
  if (nRatings > 2) {
    std::uniform_int_distribution<int> dist(0, nRatings/2);
    for (int k = 0; k < maxTriplets; k++) {
      int nTry = 0;
      bool isValid = false;
      while (nTry < 10 && !isValid) {
        nTry++;
        //sample item from first half
        ind = dist(mt);
        int item1 = itemRatings[ind].first;
        float rating1 = itemRatings[ind].second;

        //sample item from second half
        ind = dist(mt);
        secInd = nRatings/2 + ind;
        if (secInd <= nRatings-1) {
          int item2 = itemRatings[secInd].first;
          float rating2 = itemRatings[secInd].second;
          if (rating2 > rating1) {
            if (invertedPairs.count(item2) == 0) {
              invertedPairs[item2] = std::unordered_set<int>();  
            }
            invertedPairs[item2].insert(item1);
            isValid = true;
          } else if (rating1 > rating2) {
            if (invertedPairs.count(item1) == 0) {
              invertedPairs[item1] = std::unordered_set<int>();  
            }
            invertedPairs[item1].insert(item2);
            isValid = true;
          }
        } 
      }
    }
  } else if (nRatings == 2) {
    int item1 = itemRatings[0].first;
    float rating1 = itemRatings[0].second;

    int item2 = itemRatings[1].first;
    float rating2 = itemRatings[1].second;

    if (rating2 > rating1) {
      if (invertedPairs.count(item2) == 0) {
        invertedPairs[item2] = std::unordered_set<int>();  
      }
      invertedPairs[item2].insert(item1);
    } else if (rating1 > rating2) {
      if (invertedPairs.count(item1) == 0) {
        invertedPairs[item1] = std::unordered_set<int>();  
      }
      invertedPairs[item1].insert(item2);
    }

  }

  return invertedPairs;
}


std::vector<UserSets> merge(std::vector<UserSets>& a, std::vector<UserSets>& b) {
 
  std::vector<UserSets> mergeSets;

  for (auto&& uSet: a) {
    int user = uSet.user;
    auto search = std::find_if(b.begin(), b.end(), [user](UserSets const& uSet){
        return uSet.user == user;
        });
    
    if (search == b.end()) {
      continue;
    }
    
    mergeSets.push_back(uSet + (*search));
  }
  
  return mergeSets;
}


int sampleNegItem(gk_csr_t *mat, int u, float r_ui, std::mt19937& mt) {
  
  int nRatedItems = mat->rowptr[u+1] - mat->rowptr[u];
  std::uniform_int_distribution<int> dist(0, nRatedItems-1);
  std::uniform_int_distribution<int> dist2(0, mat->ncols-1);
  
  int j = -1, nTry = 0, rInd, startInd;
  float r_uj;
  int start, end;

  while (nTry < 100) {
    rInd = dist(mt);
    startInd = mat->rowptr[u];
    j = mat->rowind[startInd + rInd];
    r_uj = mat->rowval[startInd + rInd];

    if (r_uj < r_ui) {
      //found an item rated explicitly low
      break;
    } else {
      //find an implicit 0
      if (0 == rInd) {
        start = 0;
        end = j; //first rated item
      } else if (nRatedItems - 1 == rInd) {
        start = j + 1; //item appearing after last rated items
        end = mat->ncols;
      } else {
        start = j + 1; //item after j
        end = mat->rowind[startInd + rInd  + 1]; //item rated after jth item
      }
    }

    if (end - start > 0) {
      j = dist2(mt)%(end - start) + start;
    }

    nTry++;
  }
 
  if (100 == nTry) {
    j = -1;
  }

  return j;
}


//sample neg item with rating <= thresh
int sampleNegItem(gk_csr_t *mat, int u, float r_ui, std::mt19937& mt,
    float thresh) {
  
  int nRatedItems = mat->rowptr[u+1] - mat->rowptr[u];
  std::uniform_int_distribution<int> dist(0, nRatedItems-1);
  std::uniform_int_distribution<int> dist2(0, mat->ncols-1);
  
  int j = -1, nTry = 0, rInd, startInd;
  float r_uj;
  int start, end;

  while (nTry < 100) {
    rInd = dist(mt);
    startInd = mat->rowptr[u];
    j = mat->rowind[startInd + rInd];
    r_uj = mat->rowval[startInd + rInd];

    if (r_uj < r_ui && r_uj <= thresh) {
      //found an item rated explicitly low
      break;
    } else {
      //find an implicit 0
      if (0 == rInd) {
        start = 0;
        end = j; //first rated item
      } else if (nRatedItems - 1 == rInd) {
        start = j + 1; //item appearing after last rated items
        end = mat->ncols;
      } else {
        start = j + 1; //item after j
        end = mat->rowind[startInd + rInd  + 1]; //item rated after jth item
      }
    }

    if (end - start > 0) {
      j = dist2(mt)%(end - start) + start;
    }

    nTry++;
  }
 
  if (100 == nTry) {
    j = -1;
  }

  return j;
}


std::pair<int, int> samplePosNegItem(gk_csr_t *mat, int u, std::mt19937& mt,
    float thresh) {
  
  int j = -1, i = -1, posTry = 0, negTry = 0;
  float r_uj, r_ui;
  int start, end;
  int nRatedItems = mat->rowptr[u+1] - mat->rowptr[u];
  
  if (nRatedItems < 2) {
    return std::make_pair(i,j);
  }

  std::uniform_int_distribution<int> dist(0, nRatedItems-1);
  std::uniform_int_distribution<int> dist2(0, mat->ncols-1);

  //sample pos Item
  while (posTry < 100) {
    int rInd = dist(mt);
    int startInd = mat->rowptr[u];
    i = mat->rowind[startInd + rInd];
    r_ui = mat->rowval[startInd + rInd];
    if (r_ui > thresh) {
      break;
    }
    posTry++;
  }
  
  if (r_ui <= thresh) {
    //sequentially select the first r_ui > thresh
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      i = mat->rowind[ii];
      r_ui = mat->rowval[ii];
      if (r_ui > thresh) {
        break;
      }
    }
  }

  if (r_ui <= thresh) {
    i == -1;
  }

  //sample neg Item
  while (negTry < 100) {
    int rInd = dist(mt);
    int startInd = mat->rowptr[u];
    j = mat->rowind[startInd + rInd];
    r_uj = mat->rowval[startInd + rInd];

    if (r_uj <= thresh) {
      //found an item rated explicitly low
      break;
    } else {
      //find an implicit 0
      if (0 == rInd) {
        start = 0;
        end = j; //first rated item
      } else if (nRatedItems - 1 == rInd) {
        start = j + 1; //item appearing after last rated items
        end = mat->ncols;
      } else {
        start = j + 1; //item after j
        end = mat->rowind[startInd + rInd  + 1]; //item rated after jth item
      }
    }

    if (end - start > 0) {
      j = dist2(mt)%(end - start) + start;
    }

    negTry++;
  }
  
  if (100 == negTry) {
    j = -1;
  }

  return std::make_pair(i,j);
}


bool checkIf0InCSR(gk_csr_t *mat) {
  for (int u = 0; u < mat->nrows; u++) {
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      if (0 == mat->rowval[ii]) {
        return true;
      }
    }   
  }
  return false;
}


int checkIfSetsMatDiffer(std::vector<UserSets>& uSets, gk_csr_t *mat) {
  std::unordered_set<int> uItems;
  int missedInSet = 0;
  int missedInMat = 0;

  for (auto&& uSet: uSets) {
    int u = uSet.user;
    uItems.clear();
    
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      uItems.insert(item);
      if (uSet.items.find(item) == uSet.items.end()) {
        //item in matrix not present in set
        missedInSet++;
      }
    }

    for (auto&& item: uSet.items) {
      if (uItems.find(item) == uItems.end()) {
        missedInMat++;
      }
    }

  }
  
  std::cout << "missedInMat: " << missedInMat << " missedInSet: " << missedInSet
    << std::endl;

  return missedInMat + missedInSet;
}


std::unordered_set<int> checkIfSetsMatOverlap(std::vector<UserSets>& uSets, 
    gk_csr_t *mat) {
  
  int foundInSet = 0;
  std::unordered_set<int> overlapUsers;
  for (auto&& uSet: uSets) {
    int u = uSet.user;
    
    for (int ii = mat->rowptr[u]; ii < mat->rowptr[u+1]; ii++) {
      int item = mat->rowind[ii];
      if (uSet.items.find(item) != uSet.items.end()) {
        foundInSet++;
        std::cout << "u: " << u << " item: " << item 
          << " rat: " << mat->rowval[ii] << std::endl;
        overlapUsers.insert(u);
      }
    }

  }
  
  std::cout << " overlap: " << foundInSet << std::endl;

  return overlapUsers;
}


int getNNZ(gk_csr_t *mat) {
  int nnz = 0;
  for (int u = 0; u < mat->nrows; u++) {
    nnz += mat->rowptr[u+1] - mat->rowptr[u];
  }
  return nnz;
}


