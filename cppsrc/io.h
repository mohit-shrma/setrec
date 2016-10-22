#ifndef _IO_H_
#define _IO_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <Eigen/Dense>
#include "UserSets.h"
#include "util.h"

std::vector<UserSets> readSets(const char* fileName);
void writeSets(std::vector<UserSets> uSets, const char* opFName);
std::vector<int> readVector(const char *ipFileName);
std::vector<float> readFVector(const char *ipFileName);
bool isFileExist(const char *fileName);
void readEigenMat(const char* fileName, Eigen::MatrixXf& mat, int nrows, 
    int ncols);
void readEigenVec(const char* fileName, Eigen::VectorXf& vec, int nrows);
template <typename Iter>
void writeContainer(Iter it, Iter end, const char *opFileName) {
  std::ofstream opFile(opFileName);
  if (opFile.is_open()) {
    for (; it != end; ++it) {
      opFile << *it << std::endl;
    }
    opFile.close();
  }
}

template <typename Iter>
void dispContainer(Iter it, Iter end) {
    for (; it != end; ++it) {
      std::cout << *it << std::endl;
    }
}
void writeItemRMSEFreq(std::map<int, int>& itemFreq, 
    std::map<int, float>& itemRMSE, const char *opFName);
void statSets(std::vector<UserSets>& uSets);
void writeSubSampledMat(gk_csr_t *mat,  const char* sampFileName, 
    float sampPc, int seed);
#endif
