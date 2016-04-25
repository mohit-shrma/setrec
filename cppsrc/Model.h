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
    
    float estItemRating(int user, int item);
    virtual float estSetRating(int user, std::vector<int>& items) {
      std::cerr << "Base class: estSetRating not defined" << std::endl;
      return -1;
    }
    float objective(std::vector<UserSets>& uSets);
    virtual void train(const Data& data, Model& bestModel) {
      std::cerr << "Base class: train not defined" << std::endl;
    }
    float rmse(std::vector<UserSets>& uSets);
    std::string modelSign();
    void save(std::string opPrefix);
};


#endif
