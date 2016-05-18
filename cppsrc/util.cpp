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




