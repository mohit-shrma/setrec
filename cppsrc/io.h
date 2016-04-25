#ifndef _IO_H_
#define _IO_H_

#include <vector>
#include <iostream>
#include <fstream>
#include "UserSets.h"

std::vector<UserSets> readSets(const char* fileName);
void writeSets(std::vector<UserSets> uSets, const char* opFName);

#endif
