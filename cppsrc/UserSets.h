#ifndef _USER_SETS_H_
#define _USER_SETS_H_

#include "mathUtil.h"

#include <tuple>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <map>

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

    //scale score to sigmoid
    void scaleToSigm(float u_m, float g_k) {
      for (auto&& itemSet: itemSets) {
        auto score = itemSet.second;
        itemSet.second = sigmoid(score - u_m, g_k);
      } 
    }
};

#endif
