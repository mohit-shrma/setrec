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



