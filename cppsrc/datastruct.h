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
    float u_mReg;
    float g_kReg;
    float learnRate;
    float constWt;

    char *trainSetFile;
    char *testSetFile;
    char *valSetFile;
    char *ratMatFile;
    char *prefix;

    Params(int nUsers, int nItems, int facDim, int maxIter, int seed,
        float uReg, float iReg, float u_mReg, float g_kReg, float learnRate, 
        float constWt, 
        char *trainSetFile, char *testSetFile, char *valSetFile, 
        char *ratMatFile, char *prefix)
      : nUsers(nUsers), nItems(nItems), facDim(facDim), maxIter(maxIter), 
      seed(seed), 
      uReg(uReg), iReg(iReg), u_mReg(u_mReg), g_kReg(g_kReg), learnRate(learnRate), constWt(constWt),
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
      std::cout << "\nu_mReg: " << u_mReg;
      std::cout << "\ng_kReg: " << g_kReg;
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
   
    std::unordered_set<int> trainItems;
    std::unordered_set<int> trainUsers;

    int nTrainSets;
    int nTestSets;
    int nValSets;

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
        std::cout << "No. of train users: " << trainSets.size() << std::endl;
        nTrainSets = 0;
        for (auto&& uSet: trainSets) {
          nTrainSets += uSet.itemSets.size();
          for (auto&& itemSet: uSet.itemSets) {
            for (auto&& item: itemSet.first) {
              trainItems.insert(item);
            }
          }
          trainUsers.insert(uSet.user);
        }
        std::cout << "nTrainSets: " << nTrainSets << std::endl;
        std::cout << "nTrainUsers: " << trainUsers.size() << std::endl;
        std::cout << "nTrainItems: " << trainItems.size() << std::endl;
      } 

      if (NULL != params.testSetFile) {
        testSets = readSets(params.testSetFile);
        nTestSets = 0;
        //remove test sets which contain items not present in train
        auto it = std::begin(testSets);
        while (it != std::end(testSets)) {
          (*it).removeInvalSets(trainItems);
          if ((*it).itemSets.size() == 0) {
            it = testSets.erase(it);
          } else {
            ++it;
          }
        }

        for (auto&& uSet: testSets) {
          nTestSets += uSet.itemSets.size();
        }
        std::cout << "No. of test users: " << testSets.size() << std::endl;
        std::cout << "nTestSets: " << nTestSets << std::endl;
      }

      if (NULL != params.valSetFile) {
        valSets = readSets(params.valSetFile);
        nValSets = 0;
        
        //remove val sets which contain items not present in train
        auto it = std::begin(valSets);
        while (it != std::end(valSets)) {
          (*it).removeInvalSets(trainItems);
          if ((*it).itemSets.size() == 0) {
            it = valSets.erase(it);
          } else {
            ++it;
          }
        }

        for (auto&& uSet: valSets) {
          nValSets += uSet.itemSets.size();
        }
        std::cout << "No. of val users: " << valSets.size() << std::endl;
        std::cout << "nValSets: " << nValSets << std::endl;
      }

    }


    void scaleSetsToSigm(std::vector<UserSets> uSets, 
        std::map<int, float> uMidps, float g_k) {
      for (auto&& uSet: uSets) {
        int user = uSet.user;
        if (uMidps.find(user) != uMidps.end()) {
          //found user midp
          uSet.scaleToSigm(uMidps[user], g_k);
        }
      }
    }

    void scaleSetsTo01(std::vector<UserSets> uSets, 
        float maxRat) {
      for (auto&& uSet: uSets) {
        uSet.scaleTo01(maxRat);
      }
    }

    ~Data() {
      if (NULL != ratMat) {
        gk_csr_Free(&ratMat);
      }
    }

};


#endif
