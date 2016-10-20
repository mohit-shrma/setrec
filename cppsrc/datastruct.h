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
    int isMixRat;
    float rhoRMS;
    float constWt;

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
        int isMixRat, float rhoRMS,
        char *trainSetFile, char *testSetFile, char *valSetFile, 
        char *ratMatFile, char *partTrainMatFile, char *partTestMatFile, 
        char *partValMatFile, char *prefix)
      : nUsers(nUsers), nItems(nItems), facDim(facDim), maxIter(maxIter), seed(seed), 
      uReg(uReg), iReg(iReg), u_mReg(u_mReg), 
      uBiasReg(uBiasReg), iBiasReg(iBiasReg),
      g_kReg(g_kReg), learnRate(learnRate), 
      isMixRat(isMixRat), rhoRMS(rhoRMS),
      trainSetFile(trainSetFile), testSetFile(testSetFile), 
      valSetFile(valSetFile), ratMatFile(ratMatFile), 
      partTrainMatFile(partTrainMatFile), partTestMatFile(partTestMatFile),
      partValMatFile(partValMatFile), prefix(prefix) {constWt = 0;}

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
      std::cout << "\nu_mReg (uSetBiasReg): " << u_mReg;
      std::cout << "\ng_kReg (gBiasReg): " << g_kReg;
      std::cout << "\nlearnRate: " << learnRate;
      std::cout << "\nisMixRat: " << isMixRat;
      std::cout << "\nrhoRMS (gamma): " << rhoRMS;
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

    std::vector<UserSets> testValMergeSets;
    std::vector<UserSets> allSets;


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
    
    std::map<int, std::map<int, float>> valHiURatings;
    std::map<int, std::map<int, float>> testHiURatings;

    //will contain triplets u, i, j such that r_ui > r_uj
    std::vector<std::tuple<int, int, int>> allTriplets;

    int nUsers, nItems;
    
    char* prefix;

    std::vector<std::vector<std::pair<int, float>>> testValRatings;

    Data(const Params& params);

    void scaleSetsToSigm(std::vector<UserSets> uSets, 
        std::map<int, float> uMidps, float g_k);
 
    void scaleSetsTo01(float maxRat);

    void computeSetsEntropy();

    void removeInvalUI();

    void initRankMap(int seed);

    void writeTrainSetsEntropy();

    ~Data() {
      if (NULL != ratMat) {
        gk_csr_Free(&ratMat);
      }
      if (NULL != partTrainMat) {
        gk_csr_Free(&partTrainMat);
      }
      if (NULL != partTestMat) {
        gk_csr_Free(&partTestMat);
      }
      if (NULL != partValMat) {
        gk_csr_Free(&partValMat);
      }
    }

};


#endif
