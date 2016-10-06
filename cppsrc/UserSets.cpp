#include "UserSets.h"


//remove sets not containing valid items
void UserSets::removeInvalSets(std::unordered_set<int>& validItems) {
  //get the invalid items for the user
  std::unordered_set<int> invalItems;
  for (auto&& item: items) {
    if (validItems.find(item) == validItems.end()) {
      //invalid item
      invalItems.insert(item);
    }
  }
  
  //remove sets containing invalid items
  auto it = std::begin(itemSets);
  while (it != std::end(itemSets) && invalItems.size()) {
    bool isInv = false;
    for (auto&& item: (*it).first) {
      if (invalItems.find(item) != invalItems.end()) {
        //invalid item found
        isInv = true;
        break;
      }
    }
    if (isInv) {
      it = itemSets.erase(it);
    } else {
      ++it;
    }
  }

  //reassign items after removal from sets
  if (invalItems.size()) {
    items.clear();
    for(auto&& itemSet: itemSets) {
      for (auto&& item: itemSet.first) {
        items.insert(item);
      }
    }
  }

}


float UserSets::getAvgRating(std::vector<int>& items, gk_csr_t *mat) {
  float avgRat = 0.0;
  size_t found = 0;
  for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
    int item = mat->rowind[ii];
    if (std::find(items.begin(), items.end(), item) != items.end()) {
      avgRat += mat->rowval[ii];
      found += 1;
    }
    if (found == items.size()) {
      break;
    }
  }

  if (found != items.size()) {
    std::cerr << "items in set not found in user-item ratings" << std::endl;
  }

  avgRat = avgRat/items.size();

  return avgRat;
}


//remove sets which deviate from average rating by +- 0.5
void UserSets::removeOverUnderRatedSets(gk_csr_t *mat) {
  auto it = std::begin(itemSets);
  while (it != std::end(itemSets)) {
    float avgRat = getAvgRating((*it).first, mat);
    if (fabs(avgRat - (*it).second) >= 0.5) {
      //over-under rated set, remove it from training
      it = itemSets.erase(it);
    } else {
      ++it;
    }
  }
}


//scale score to sigmoid
void UserSets::scaleToSigm(float u_m, float g_k) {
  for (auto&& itemSet: itemSets) {
    auto score = itemSet.second;
    itemSet.second = sigmoid(score - u_m, g_k);
  } 
}


//scale score to 0 - 1
void UserSets::scaleTo01(float maxRat) {
  for (auto&& itemSet: itemSets) {
    itemSet.second = itemSet.second/maxRat;
  }
}


//sample pos, neg set ind
std::pair<int, int> UserSets::sampPosNeg(std::mt19937& mt) {
  std::uniform_int_distribution<int> dist(0, itemSets.size()-1);
  
  //sample first set
  int firstInd = dist(mt);
  int secondInd = -1;
  int highInd = -1, lowInd = -1;

  for (int i = 0; i < 50; i++) {
    //sample second set
    secondInd = dist(mt);
    if (itemSets[secondInd].second != itemSets[firstInd].second) {
      break;
    }
  }
  
  if (itemSets[secondInd].second == itemSets[firstInd].second) {
    //manual search for high ind
    std::vector<int> setInds(itemSets.size());
    std::iota(setInds.begin(), setInds.end(), 0);
    std::shuffle(setInds.begin(), setInds.end(), mt);

    for (auto i : setInds) {
      if (itemSets[i].second != itemSets[firstInd].second) {
        secondInd = i;
        break;
      } 
    }

  }
  
  if (itemSets[secondInd].second != itemSets[firstInd].second) {
    if (itemSets[firstInd].second > itemSets[secondInd].second) {
      highInd = firstInd;
      lowInd = secondInd;
    } else {
      highInd = secondInd;
      lowInd = firstInd;
    }
  }

  return std::make_pair(highInd, lowInd);
}


//sample sets s,t such that r_us <= lb, r_ut > lb
std::pair<int, int> UserSets::sampPosNeg(std::mt19937& mt, int lb) {
  std::uniform_int_distribution<int> dist(0, itemSets.size()-1);
  
  //sample first set
  int firstInd = dist(mt);
  int secondInd = -1;
  int highInd = -1, lowInd = -1;

  for (int i = 0; i < 50; i++) {
    //sample second set
    secondInd = dist(mt);
    if (itemSets[firstInd].second  <= lb) {
      if (itemSets[secondInd].second > lb) {
        break;
      }
    } else if (itemSets[firstInd].second  > lb) {
      if (itemSets[secondInd].second <= lb) {
        break;
      }
    }
  }
  
  if (itemSets[secondInd].second == itemSets[firstInd].second) {
    //manual search for second
    std::vector<int> setInds(itemSets.size());
    std::iota(setInds.begin(), setInds.end(), 0);
    std::shuffle(setInds.begin(), setInds.end(), mt);
    for (auto i : setInds) {
      if (itemSets[firstInd].second <= lb) {
        if (itemSets[i].second > lb)  {
          secondInd = i;
          break;
        }
      } else if (itemSets[firstInd].second > lb) {
        if (itemSets[i].second <= lb) {
          secondInd = i;
          break;
        }
      }
    }
  }
 
  if ((itemSets[firstInd].second <= lb && itemSets[secondInd].second > lb)
      || (itemSets[firstInd].second > lb && itemSets[secondInd].second <= lb)) {
    if (itemSets[firstInd].second > itemSets[secondInd].second) {
      highInd = firstInd;
      lowInd = secondInd;
    } else {
      highInd = secondInd;
      lowInd = firstInd;
    }
  }

  return std::make_pair(highInd, lowInd);
}


UserSets UserSets::operator+(const UserSets& b) {
  std::vector<std::pair<std::vector<int>, float>> combItemSets;
  UserSets uSet;  
  
  if (user == b.user) {
    for (auto&& itemSet: b.itemSets) {
      combItemSets.push_back(itemSet);
    }
    for (auto&& itemSet: itemSets) {
      combItemSets.push_back(itemSet);
    }
   uSet = UserSets(user, combItemSets);
  }
  return uSet;
}


void UserSets::computeEntropy(gk_csr_t *mat) {

  setsEntropy.clear();

  std::map<int, float> itemRatings;
  for (int ii = mat->rowptr[user]; ii < mat->rowptr[user+1]; ii++) {
    int item = mat->rowind[ii];
    float rating = mat->rowval[ii];
    itemRatings[item] = rating;
  }

  //bins for rating: [0,1), [1,2), [2,3), [3,4), [4,5)
  for (int i = 0; i < itemSets.size(); i++) {
    
    auto& itemsSet = itemSets[i].first;
    float entropy = 0;
    
    std::vector<float> bins(5, 0.0);
    
    for (auto&& item: itemsSet) {
      float rating = itemRatings[item];
      if (rating < 1) {
        bins[0] += 1;
      } else if (rating < 2) {
        bins[1] += 1;
      } else if (rating < 3) {
        bins[2] += 1;
      } else if (rating < 4) {
        bins[3] += 1;
      } else {
        bins[4] += 1;
      }
    }
    
    for (auto&& bin: bins) {
      if (bin > 0) {
        entropy += -(bin/itemsSet.size()) * std::log10(bin/itemsSet.size());
      }
    }
     
    setsEntropy.push_back(entropy);
  }

}


void UserSets::orderSetsByEntropy() {
  std::vector<std::pair<ItemsSetNRating, double>> setsNEntropy;
  for (int i = 0; i < itemSets.size(); i++) {
    setsNEntropy.push_back(std::make_pair(itemSets[i], setsEntropy[i]));
  }

  //sort in ascending order
  std::sort(setsNEntropy.begin(), setsNEntropy.end(), 
      [](const std::pair<ItemsSetNRating, double> &a, 
        const std::pair<ItemsSetNRating, double> &b){
        return a.second < b.second;
      });
  
  //update sets
  itemSets.clear();
  setsEntropy.clear();
  
  for (auto&& setNEntropy: setsNEntropy) {
    itemSets.push_back(setNEntropy.first);
    setsEntropy.push_back(setNEntropy.second);
  }
  
}


void UserSets::removeHighEntropy(float pc) {
  //orer sets by entropy in ascending order
  orderSetsByEntropy();
  int nSets = itemSets.size();
  int nRemSets = nSets*pc;
  std::unordered_set<int> validItems;
  for (int i = 0; i < nSets-nRemSets; i++) {
    auto& itemsSetNRating = itemSets[i];
    for (auto&& item: itemsSetNRating.first) {
      validItems.insert(item);
    }
  }
  removeInvalSets(validItems);
}


void UserSets::removeLowEntropy(float pc) {
  orderSetsByEntropy();
  int nSets = itemSets.size();
  int nRemSets = nSets*pc;
  std::unordered_set<int> validItems;
  for (int i = nSets-1; i >= nRemSets; i--) {
    auto& itemsSetNRating = itemSets[i];
    for (auto&& item: itemsSetNRating.first) {
      validItems.insert(item);
    }
  }
  removeInvalSets(validItems);
}


void UserSets::removeRandom(float pc, int seed) {
  std::mt19937 mt(seed); 
  orderSetsByEntropy();
  int nSets = itemSets.size();
  std::uniform_int_distribution<> dis(0, nSets-1);
  int nRemSets = nSets*pc;
  std::unordered_set<int> validItems;
  std::unordered_set<int> remSetsInd;
  int nTry = 0;
  while (nTry < 100+nRemSets && remSetsInd.size() < nRemSets) {
    int setInd = dis(mt);
    remSetsInd.insert(setInd);
    nTry++;
  }
  
  for (int i = 0; i < nSets; i++) {
    if (remSetsInd.count(i) > 0 ) {
      continue;
    }
    auto& itemsSetNRating = itemSets[i];
    for (auto&& item: itemsSetNRating.first) {
      validItems.insert(item);
    }
  }

  removeInvalSets(validItems);
  

}


