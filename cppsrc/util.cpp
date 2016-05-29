#include "util.h"


bool descComp(std::pair<int, float>& a, std::pair<int, float>& b) {
  return a.second > b.second;
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



