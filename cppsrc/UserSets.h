#ifndef _USER_SETS_H_
#define _USER_SETS_H_

#include <iostream>
#include <vector>
#include <set>
#include <map>

class UserSets {

  public:
    int user;
    std::vector<std::vector<int>> itemSets;
    std::vector<float> setScores;
    std::unordered_set<int> items;
    std::map<int, int> item2SetInd;

    //constructor
    UserSets(int user, std::vector<std::vector<int>> itemSets, 
        std::vector<float> setScores)
      :user(user), itemSets(itemSets), setScores(setScores) {
        //get items
        for (size_t i = 0; i < itemSets.size(); i++) {
          auto itemSet = itemSets[i];
          for (auto&& item: itemSet) {
            //check if item occur in multiple sets for the user
            if (uItemsSet.find(item) != uItemsSet.end()) {
              //found item across multiple sets for user
              std::cerr << "\n!!! found item again: " << item << " " << user
                << std::endl;
            }
            items.insert(item);
            item2SetInd[item] = i;
          }
        }
      
      }

}

#endif
