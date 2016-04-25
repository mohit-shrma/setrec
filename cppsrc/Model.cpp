#include "Model.h"

Model::Model(const Params &params) {
  
  nUsers    = params.nUsers;
  nItems    = params.nItems;
  facDim    = params.facDim;
  uReg      = params.uReg;
  iReg      = params.iReg;
  learnRate = params.learnRate;

  //random engine
  std::mt19937 mt(params.seed);
  std::uniform_real_distribution<> dis(0, 1);

  //initialize User factors
  U = Eigen::MatrixXf(nUsers, facDim);
  for (int u = 0; u < nUsers; u++) {
    for (int k = 0; k < facDim; k++) {
      U(u, k) = dis(mt);
    }
  }

  //initialize item factors
  V = Eigen::MatrixXf(nItems, facDim);
  for (int item = 0; item < nItems; item++) {
    for (int k = 0; k < facDim; k++) {
      V(item, k) = dis(mt);
    }
  }

}

