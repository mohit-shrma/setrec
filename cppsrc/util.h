#ifndef _UTIL_H_
#define _UTIL_H_

#include <tuple>
#include <unordered_set>
#include <vector>
#include <map>

#include "UserSets.h"
#include "GKlib.h"

bool descComp(std::pair<int, float>& a, std::pair<int, float>& b);
void removeInvalUIFrmSets(std::vector<UserSets>& uSets, 
    std::unordered_set<int>& valUsers, std::unordered_set<int>& valItems);
void userItemsFrmSets(std::vector<UserSets>& uSets, 
    std::unordered_set<int>& users, std::unordered_set<int>& items);
std::map<int, int> getItemFreq(std::vector<UserSets>& uSets);
void removeOverUnderRatedSets(std::vector<UserSets>& uSets, gk_csr_t* ratMat);
void removeSetsWOValItems(std::vector<UserSets>& uSets, 
    std::unordered_set<int>& valItems);
void removeSetsWOVal(std::vector<UserSets>& uSets,
    std::unordered_set<int>& valUsers, std::unordered_set<int>& valItems);

#endif
