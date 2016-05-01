#ifndef _UTIL_H_
#define _UTIL_H_

#include <tuple>
#include <unordered_set>
#include <vector>

#include "UserSets.h"

bool descComp(std::pair<int, float>& a, std::pair<int, float>& b);
void removeInvalUIFrmSets(std::vector<UserSets>& uSets, 
    std::unordered_set<int>& valUsers, std::unordered_set<int>& valItems);
void userItemsFrmSets(std::vector<UserSets>& uSets, 
    std::unordered_set<int>& users, std::unordered_set<int>& items);

#endif
