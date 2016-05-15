#ifndef _USER_SETS_H_
#define _USER_SETS_H_

#include "mathUtil.h"

#include <random>
#include <tuple>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <map>
#include "GKlib.h"

class UserSets {

  public:
    int user;
    std::vector<std::pair<std::vector<int>, float>> itemSets;
    std::unordered_set<int> items;
    std::map<int, int> item2SetInd;

    //constructor
    UserSets(int user, std::vector<std::pair<std::vector<int>,float>> itemSets)
      :user(user), itemSets(itemSets) {
        //get items
        for (size_t i = 0; i < itemSets.size(); i++) {
          auto itemSet = itemSets[i].first;
          for (auto&& item: itemSet) {
            //check if item occur in multiple sets for the user
            if (items.find(item) != items.end()) {
              //found item across multiple sets for user
              //std::cerr << "\n!!! found item again: " << item << " " << user
              //  << std::endl;
            }
            items.insert(item);
            item2SetInd[item] = i;
          }
        }
      
      }
   
    //remove sets not containing valid items
    void removeInvalSets(std::unordered_set<int>& validItems) {
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
    
    float getAvgRating(std::vector<int>& items, gk_csr_t *mat) {
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
    void removeOverUnderRatedSets(gk_csr_t *mat) {
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
    void scaleToSigm(float u_m, float g_k) {
      for (auto&& itemSet: itemSets) {
        auto score = itemSet.second;
        itemSet.second = sigmoid(score - u_m, g_k);
      } 
    }

    //scale score to 0 - 1
    void scaleTo01(float maxRat) {
      for (auto&& itemSet: itemSets) {
        itemSet.second = itemSet.second/maxRat;
      }
    }
    
    //sample pos, neg set ind
    std::pair<int, int> sampPosNeg(std::mt19937 mt) {
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
        //manual search for low, high ind
        for (size_t i = 0; i < itemSets.size(); i++) {
          if (itemSets[i].second != itemSets[firstInd].second) {
            secondInd = i;
          } else if (itemSets[i].second != itemSets[secondInd].second) {
            firstInd = i;
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

};

#endif
