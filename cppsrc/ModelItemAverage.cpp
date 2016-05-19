#include "ModelItemAverage.h"


float ModelItemAverage::estItemRating(int user, int item) {
  //return gBias;
  if (globalItemRatings.find(item) == globalItemRatings.end()) {
    std::cerr << "Not found: " << item << std::endl;
    //return gBias;
  }
  return globalItemRatings[item];
}


void ModelItemAverage::train(const Data& data, const Params& params,
    Model& bestModel) {
  int nnz = 0;
  std::map<int, int> globalItemCount;
  gBias = 0;
  for (int u = 0; u < data.partTrainMat->nrows; u++) {
    for (int ii = data.partTrainMat->rowptr[u]; 
        ii < data.partTrainMat->rowptr[u+1]; u++) {
      int item = data.partTrainMat->rowind[ii];
      float r_ui = data.partTrainMat->rowval[ii];
      gBias += r_ui;
      nnz++;
      if (globalItemRatings.find(item) == globalItemRatings.end()) {
        globalItemRatings[item] = 0;
        globalItemCount[item] = 0;
      }
      globalItemCount[item] += 1;
      globalItemRatings[item] += r_ui;
    }
  }
  gBias = gBias/nnz;

  for (auto&& kv: globalItemRatings) {
    int item = kv.first;
    globalItemRatings[item] = globalItemRatings[item]/globalItemCount[item];
  }
  
  bestModel = *this;
}

