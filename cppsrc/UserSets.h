#ifndef _USER_SETS_H_
#define _USER_SETS_H_

#include "mathUtil.h"

#include <cmath>
#include <random>
#include <tuple>
#include <iostream>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <map>
#include "GKlib.h"

using ItemsSet = std::vector<int>;
using ItemsSetNRating = std::pair<ItemsSet, float>;

class UserSets {

  public:
    int user;
    std::vector<ItemsSetNRating> itemSets;
    std::vector<double> setsEntropy;
    std::unordered_set<int> items;

    //constructor
    UserSets() {
      user = -1;
    }

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
          }
        }
      
      } 
   
    //remove sets not containing valid items
    void removeInvalSets(std::unordered_set<int>& validItems);
   
    float getAvgRating(std::vector<int>& items, gk_csr_t *mat);

    //remove sets which deviate from average rating by +- 0.5
    void removeOverUnderRatedSets(gk_csr_t *mat);

    //scale score to sigmoid
    void scaleToSigm(float u_m, float g_k);

    //scale score to 0 - 1
    void scaleTo01(float maxRat);
    
    //sample pos, neg set ind
    std::pair<int, int> sampPosNeg(std::mt19937& mt);

    //sample sets s,t such that r_us <= lb, r_ut > lb
    std::pair<int, int> sampPosNeg(std::mt19937& mt, int lb) ;
         
    UserSets operator+(const UserSets& b);
    
    void computeEntropy(gk_csr_t* mat);

    void orderSetsByEntropy();
    void removeHighEntropy(float pc);
    void removeLowEntropy(float pc);
    void removeRandom(float pc, int seed);

    float getVarPickiness(gk_csr_t *mat) const;
};

#endif
