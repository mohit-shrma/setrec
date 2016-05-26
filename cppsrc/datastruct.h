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
    float uBiasReg;
    float iBiasReg;
    float g_kReg;
    float learnRate;
    float constWt;
    float rhoRMS;

    char *trainSetFile;
    char *testSetFile;
    char *valSetFile;
    char *ratMatFile;
    char *partTrainMatFile;
    char *partTestMatFile;
    char *partValMatFile;
    char *prefix;

    Params(int nUsers, int nItems, int facDim, int maxIter, int seed,
        float uReg, float iReg, float u_mReg, 
        float uBiasReg, float iBiasReg,
        float g_kReg, float learnRate,
        float constWt, float rhoRMS,
        char *trainSetFile, char *testSetFile, char *valSetFile, 
        char *ratMatFile, char *partTrainMatFile, char *partTestMatFile, 
        char *partValMatFile, char *prefix)
      : nUsers(nUsers), nItems(nItems), facDim(facDim), maxIter(maxIter), seed(seed), 
      uReg(uReg), iReg(iReg), u_mReg(u_mReg), 
      uBiasReg(uBiasReg), iBiasReg(iBiasReg),
      g_kReg(g_kReg), learnRate(learnRate), 
      constWt(constWt), rhoRMS(rhoRMS),
      trainSetFile(trainSetFile), testSetFile(testSetFile), 
      valSetFile(valSetFile), ratMatFile(ratMatFile), 
      partTrainMatFile(partTrainMatFile), partTestMatFile(partTestMatFile),
      partValMatFile(partValMatFile), prefix(prefix) {}

    void display() {
      std::cout << "******* PARAMETERS ********";
      std::cout << "\nnUsers: " << nUsers;
      std::cout << "\nnItems: " << nItems;
      std::cout << "\nfacDim: " << facDim;
      std::cout << "\nmaxIter: " << maxIter;
      std::cout << "\nseed: " << seed;
      std::cout << "\nuReg: " << uReg;
      std::cout << "\niReg: " << iReg;
      std::cout << "\nuBiasReg: " << uBiasReg;
      std::cout << "\niBiasReg: " << iBiasReg;
      std::cout << "\nu_mReg: " << u_mReg;
      std::cout << "\ng_kReg: " << g_kReg;
      std::cout << "\nlearnRate: " << learnRate;
      std::cout << "\nconstWt: " << constWt;
      std::cout << "\nrhoRMS: " << rhoRMS;
      std::cout << "\ntrainSetFile: " << trainSetFile;
      std::cout << "\ntestSetFile: " << testSetFile;
      std::cout << "\nvalSetFile: " << valSetFile;
      std::cout << "\nratMatFile: " << ratMatFile;
      std::cout << "\npartTrainMatFile: " << partTrainMatFile;
      std::cout << "\npartTestMatFile: " << partTestMatFile;
      std::cout << "\npartValMatFile: " << partValMatFile;
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
    gk_csr_t* partTrainMat;
    gk_csr_t* partTestMat;
    gk_csr_t* partValMat;

    //used for ranking evaluation
    std::map<int, int> valUItems;
    std::map<int, int> testUItems;
    std::map<int, std::unordered_set<int>> ignoreUItems;
    std::map<int, std::map<int, float>> valURatings;
    std::map<int, std::map<int, float>> testURatings;

    int nUsers, nItems;
    
    char* prefix;

    Data(const Params& params) {
      
      nUsers = params.nUsers;
      nItems = params.nItems;
      prefix = params.prefix;

      if (NULL != params.ratMatFile) {
        std::cout << "\nReading rating matrix 0 indexed..." << params.ratMatFile;
        ratMat = gk_csr_Read(params.ratMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(ratMat, GK_CSR_COL);
      }
      
      if (NULL != params.partTrainMatFile) {
        std::cout << "\nReading partial rating matrix 0 indexed..." 
          << params.partTrainMatFile;
        partTrainMat = gk_csr_Read(params.partTrainMatFile, GK_CSR_FMT_CSR, 1, 0);
        gk_csr_CreateIndex(partTrainMat, GK_CSR_COL);
      }
      
      if (NULL != params.partTestMatFile) {
        std::cout << "\nReading partial rating matrix 0 indexed..." 
          << params.partTestMatFile;
        partTestMat = gk_csr_Read(params.partTestMatFile, GK_CSR_FMT_CSR, 1, 0);
      }
      
      if (NULL != params.partValMatFile) {
        std::cout << "\nReading partial rating matrix 0 indexed..." 
          << params.partValMatFile;
        partValMat = gk_csr_Read(params.partValMatFile, GK_CSR_FMT_CSR, 1, 0);
      }
      
      if (NULL != params.trainSetFile) {
        trainSets  = readSets(params.trainSetFile);    
        
        //remove over-under rated sets
        //removeOverUnderRatedSets(trainSets, ratMat);

        

        std::cout << "No. of train users: " << trainSets.size() << std::endl;
        nTrainSets = 0;
        for (auto&& uSet: trainSets) {
          nTrainSets += uSet.itemSets.size();
        }
        auto trainUserItems = getUserItems(trainSets);
        trainUsers = trainUserItems.first;
        trainItems = trainUserItems.second;
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


    void initRankMap(int seed) {
      std::vector<std::pair<int, float>> itemActRatings;
      for (auto&& uSet: trainSets) {
        int user = uSet.user;
        auto setItems = uSet.items;
        itemActRatings.clear();

        for (int ii = ratMat->rowptr[user]; ii < ratMat->rowptr[user+1]; ii++) {
          int item = ratMat->rowind[ii];
          if (setItems.find(item) == setItems.end() 
              && trainItems.find(item) != trainItems.end()) {
            //not found in user sets, but exist in train items
            itemActRatings.push_back(std::make_pair(item, ratMat->rowval[ii]));
          }
        }
        
        std::map<int, float> valMap;
        std::map<int, float> testMap;

        //shuffle before partitioning
        //initialize random engine
        std::mt19937 mt(seed);
        std::shuffle(itemActRatings.begin(), itemActRatings.end(), mt);

        if (itemActRatings.size() >= 4) {
          for (auto it = itemActRatings.begin(); 
              it != itemActRatings.begin() + itemActRatings.size()/2; it++) {
            valMap[(*it).first] = (*it).second;
          }
          for (auto it = itemActRatings.begin() + itemActRatings.size()/2; 
              it != itemActRatings.end(); it++) {
            testMap[(*it).first] = (*it).second;
          }
          valURatings[user]  = valMap;
          testURatings[user] = testMap;
        }

        //get top-2 elements in beginning
        std::nth_element(itemActRatings.begin(), itemActRatings.begin()+(2-1),
            itemActRatings.end(), descComp);
        
        if (!(itemActRatings[0].second > 3 && itemActRatings[1].second > 3)) {
          continue;
        }

        valUItems[user] = itemActRatings[0].first;
        testUItems[user] = itemActRatings[1].first;

        std::unordered_set<int> uIgnoreItems;
        for (auto it = itemActRatings.begin()+2; it != itemActRatings.end(); 
            it++) {
          if ( (*it).second > 3) {
            uIgnoreItems.insert((*it).first);
          }
        }
        ignoreUItems[user] = uIgnoreItems;
      }
    }
  

    ~Data() {
      if (NULL != ratMat) {
        gk_csr_Free(&ratMat);
      }
    }

};


#endif
