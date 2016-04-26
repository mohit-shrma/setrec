#ifndef _DATASTRUCT_H_
#define _DATASTRUCT_H_

#include <vector>

#include "GKlib.h"
#include "UserSets.h"
#include "io.h"

class Params {

  public:
    int nUsers;
    int nItems;
    int facDim;
    int maxIter;
    int seed;  
    
    float uReg;
    float iReg;
    float learnRate;
    float constWt;

    char *trainSetFile;
    char *testSetFile;
    char *valSetFile;
    char *ratMatFile;
    char *prefix;

    Params(int nUsers, int nItems, int facDim, int maxIter, int seed,
        float uReg, float iReg, float learnRate, float constWt,
        char *trainSetFile, char *testSetFile, char *valSetFile, 
        char *ratMatFile, char *prefix)
      : nUsers(nUsers), nItems(nItems), facDim(facDim), maxIter(maxIter), 
      seed(seed), 
      uReg(uReg), iReg(iReg), learnRate(learnRate), constWt(constWt),
      trainSetFile(trainSetFile), testSetFile(testSetFile), 
      valSetFile(valSetFile), ratMatFile(ratMatFile), prefix(prefix) {}

    void display() {
      std::cout << "******* PARAMETERS ********";
      std::cout << "\nnUsers: " << nUsers;
      std::cout << "\nnItems: " << nItems;
      std::cout << "\nfacDim: " << facDim;
      std::cout << "\nmaxIter: " << maxIter;
      std::cout << "\nseed: " << seed;
      std::cout << "\nuReg: " << uReg;
      std::cout << "\niReg: " << iReg;
      std::cout << "\nlearnRate: " << learnRate;
      std::cout << "\nconstWt: " << constWt;
      std::cout << "\ntrainSetFile: " << trainSetFile;
      std::cout << "\ntestSetFile: " << testSetFile;
      std::cout << "\nvalSetFile: " << valSetFile;
      std::cout << "\nratMatFile: " << ratMatFile;
      std::cout << "\nprefix: " << prefix;
    }

};


class Data {
  public:

    std::vector<UserSets> trainSets;
    std::vector<UserSets> testSets;
    std::vector<UserSets> valSets;
   
    gk_csr_t* ratMat;

    int nUsers, nItems;
    
    char* prefix;

    Data(const Params& params) {
      
      nUsers = params.nUsers;
      nItems = params.nItems;
      prefix = params.prefix;

      if (NULL != params.ratMatFile) {
        std::cout << "\nReading rating matrix 0 indexed...";
        ratMat = gk_csr_Read(params.ratMatFile, GK_CSR_FMT_CSR, 1, 0);
      }
      
      if (NULL != params.trainSetFile) {
        trainSets  = readSets(params.trainSetFile);    
        std::cout << "No. of train sets: " << trainSets.size() << std::endl;
      }

      if (NULL != params.testSetFile) {
        testSets = readSets(params.testSetFile);
      }

      if (NULL != params.valSetFile) {
        valSets = readSets(params.valSetFile);
      }

    }


    ~Data() {
      if (NULL != ratMat) {
        gk_csr_Free(&ratMat);
      }
    }

};


#endif
