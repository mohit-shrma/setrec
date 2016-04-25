#ifndef _MODEL_H_
#define _MODEL_H_

#include <Eigen/Dense>
#include <random>
#include "datastruct.h"

class Model {
  
  public:
    int nUsers;
    int nItems;
   
    //user-item latent factors
    Eigen::MatrixXf U;
    Eigen::MatrixXf V;
    
    //size of latent factors
    int facDim;
    
    //regularization
    float uReg, iReg;

    float learnRate;

    Model(const Params &params);

};


#endif
