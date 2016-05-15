#ifndef _DATASTRUCT_H_
#define _DATASTRUCT_H_

#include <vector>

#include "GKlib.h"
#include "UserSets.h"
#include "io.h"
#include "util.h"

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
    float rhoRMS;

    char *trainSetFile;
    char *testSetFile;
    char *valSetFile;
    char *ratMatFile;
    char *partMatFile;
    char *prefix;

    Params(int nUsers, int nItems, int facDim, int maxIter, int seed,
        float uReg, float iReg, float u_mReg, float g_kReg, float learnRate, 
        float constWt, float rhoRMS,
        char *trainSetFile, char *testSetFile, char *valSetFile, 
        char *ratMatFile, char *partMatFile, char *prefix)
      : nUsers(nUsers), nItems(nItems), facDim(facDim), maxIter(maxIter), seed(seed), 
      uReg(uReg), iReg(iReg), u_mReg(u_mReg), g_kReg(g_kReg), learnRate(learnRate), 
      constWt(constWt), rhoRMS(rhoRMS),
      trainSetFile(trainSetFile), testSetFile(testSetFile), 
      valSetFile(valSetFile), ratMatFile(ratMatFile), partMatFile(partMatFile), 
      prefix(prefix) {}

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
      std::cout << "\nrhoRMS: " << rhoRMS;
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

    std::unordered_set<int> invalUsers;
    std::unordered_set<int> invalItems;

    int nTrainSets;
    int nTestSets;
    int nValSets;

    //full rating matrix
    gk_csr_t* ratMat;
    
    //partial rating matrix
    gk_csr_t* partMat;

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
      
      if (NULL != params.partMatFile) {
        std::cout << "\nReading partial rating matrix 0 indexed..." << std::endl;
        partMat = gk_csr_Read(params.partMatFile, GK_CSR_FMT_CSR, 1, 0);
      }
      
      if (NULL != params.trainSetFile) {
        trainSets  = readSets(params.trainSetFile);    
        
        //remove over-under rated sets
        //removeOverUnderRatedSets(trainSets, ratMat);

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
        removeSetsWOValItems(testSets, trainItems);
        //remove over-under rated sets
        //removeOverUnderRatedSets(testSets, ratMat);

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
        removeSetsWOValItems(valSets, trainItems);
        //remove over-under rated sets
        //removeOverUnderRatedSets(valSets, ratMat);

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


    void scaleSetsTo01(float maxRat) {
      for (auto&& uSet: trainSets) {
        uSet.scaleTo01(maxRat);
      }
      for (auto&& uSet: testSets) {
        uSet.scaleTo01(maxRat);
      }
      for (auto&& uSet: valSets) {
        uSet.scaleTo01(maxRat);
      }
    }


    void removeInvalUI() {
      //set of valid items = trainItems - invalItems
      std::unordered_set<int> valItems;
      for (auto item: trainItems) {
        if (invalItems.find(item) == invalItems.end()) {
          //train item is valid
          valItems.insert(item);
        }
      }

      //set of valid users = trainUsers - invalUsers
      std::unordered_set<int> valUsers;
      for (auto user: trainUsers) {
        if (invalUsers.find(user) == invalUsers.end()) {
          //train user is valid
          valUsers.insert(user);
        }
      }
      
      removeInvalUIFrmSets(trainSets, valUsers, valItems);
      //update train users and items
      userItemsFrmSets(trainSets, trainUsers, trainItems);
      nTrainSets = trainSets.size();

      removeInvalUIFrmSets(testSets, valUsers, valItems);
      nTestSets = testSets.size();

      removeInvalUIFrmSets(valSets, valUsers, valItems);
      nValSets = valSets.size();
    }


    ~Data() {
      if (NULL != ratMat) {
        gk_csr_Free(&ratMat);
      }
    }

};


#endif
