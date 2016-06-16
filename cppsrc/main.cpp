#include <iostream>
#include <cstdlib>

#include "datastruct.h"
#include "ModelAverage.h"
#include "ModelAverageWBias.h"
#include "ModelAverageWSetBias.h"
#include "ModelBaseline.h"
#include "ModelMajority.h"
#include "ModelMajorityWCons.h"
#include "ModelAverageHinge.h"
#include "ModelAverageWPart.h"
#include "ModelBaseline.h"
#include "ModelAverageBiasesOnly.h"
#include "ModelAverageWGBias.h"
#include "ModelItemAverage.h"
#include "ModelAverageSetBiasWPart.h"

#include "ModelMFWBias.h"
#include "ModelAverageHingeWBias.h"
#include "ModelAverageLogWBias.h"
#include "ModelAverageBPRWBias.h"
#include "ModelBPR.h"
#include "ModelBPRTop.h"
#include "ModelAverageBPRWBiasTop.h"


Params parse_cmd_line(int argc, char* argv[]) {
  if (argc < 23) {
    std::cerr << "Not enough args" << std::endl;
    exit(1);
  }
  
  return Params(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]),
      std::atoi(argv[4]), std::atoi(argv[5]),
      std::atof(argv[6]), std::atof(argv[7]), std::atof(argv[8]),
      std::atof(argv[9]), std::atof(argv[10]),
      std::atof(argv[11]), std::atof(argv[12]), std::atof(argv[13]), std::atof(argv[14]),
      argv[15], argv[16], argv[17], argv[18], argv[19], argv[20], argv[21], argv[22]);
}


void loadModelNRMSEs(Data& data, const Params& params) {
  ModelAverage modelAvg(params, 
      "mf_uFac_854X12549_5_0.010000_0.010000_0.001000.mat", 
      "mf_iFac_854X12549_5_0.010000_0.010000_0.001000.mat");

  std::vector<int> invalUsers = readVector("mf_854X12549_5_0.010000_0.010000_0.001000_invalUsers.txt");
  std::vector<int> invalItems = readVector("mf_854X12549_5_0.010000_0.010000_0.001000_invalItems.txt");
 
  for (auto&& user: invalUsers) {
    data.invalUsers.insert(user);
  }
  
  for (auto&& item: invalItems) {
    data.invalItems.insert(item);
  }

  data.removeInvalUI();

  std::cout << "Train RMSE: " << modelAvg.rmse(data.trainSets) << std::endl;
  std::cout << "Train ratings RMSE: " 
    << modelAvg.rmse(data.trainSets, data.ratMat) << std::endl;
  std::cout << "Test RMSE: " << modelAvg.rmse(data.testSets) << std::endl;
  std::cout << "Test ratings RMSE: " 
    << modelAvg.rmse(data.testSets, data.ratMat) << std::endl;
  std::cout << "Val RMSE: " << modelAvg.rmse(data.valSets) << std::endl;
  std::cout << "Val ratings RMSE: " << modelAvg.rmse(data.valSets, data.ratMat)
    << std::endl;
}


int main(int argc, char *argv[]) {
  Params params = parse_cmd_line(argc, argv);
  params.display();
  Data data(params);
  
  data.initRankMap(params.seed);

  std::cout << "Train users: " << data.trainUsers.size() << " Train items: " 
    << data.trainItems.size() << std::endl;
 

  writeSubSampledMat(data.partTrainMat, 
      "train_0.1.csr", 0.1, params.seed);
  writeSubSampledMat(data.partTrainMat, 
      "train_0.25.csr", 0.25, params.seed);
  writeSubSampledMat(data.partTrainMat, 
      "train_0.5.csr", 0.5, params.seed);
  writeSubSampledMat(data.partTrainMat, 
      "train_0.75.csr", 0.75, params.seed);

  //std::string opFName = std::string(params.prefix) + "_trainSet_temp";
  //writeSets(data.trainSets, opFName.c_str());

  
  /*
  ModelMFWBias modelAvg(params);
  ModelMFWBias bestModel(modelAvg);
  modelAvg.train(data, params, bestModel);
  /* 
  std::vector<UserSets> undSets = readSets("ml_set.und.lfs");
  std::cout << "Underrated sets b4 rem stats: " << std::endl;
  statSets(undSets);
  removeSetsWOVal(undSets, data.trainUsers, data.trainItems);
  std::cout << "Underrated sets aftr rem stats: " << std::endl;
  statSets(undSets);

  std::vector<UserSets> ovrSets = readSets("ml_set.ovr.lfs");
  std::cout << "Overrated sets b4 rem stats: " << std::endl;
  statSets(ovrSets);
  removeSetsWOVal(ovrSets, data.trainUsers, data.trainItems);
  std::cout << "Overrated sets aftr rem stats: " << std::endl;
  statSets(ovrSets);

  std::vector<UserSets> allSets;
  //allSets.insert(allSets.end(), data.trainSets.begin(), data.trainSets.end());
  allSets.insert(allSets.end(), data.testSets.begin(), data.testSets.end());
  //allSets.insert(allSets.end(), data.valSets.begin(), data.valSets.end());
  allSets.insert(allSets.end(), undSets.begin(), undSets.end());
  allSets.insert(allSets.end(), ovrSets.begin(), ovrSets.end());
  
  std::cout << "All sets b4 rem stats: " << std::endl;
  statSets(allSets);
  removeSetsWOVal(allSets, data.trainUsers, data.trainItems);
  std::cout << "All sets aftr rem stats: " << std::endl;
  statSets(allSets);
  
  std::cout << "Under RMSE: " << bestModel.rmse(undSets, data.ratMat) << std::endl;
  std::cout << "Over RMSE: " << bestModel.rmse(ovrSets, data.ratMat) << std::endl;
  std::cout << "All RMSE: " << bestModel.rmse(allSets, data.ratMat) << std::endl;
  */
  /*
  float trainRMSE = bestModel.rmse(data.trainSets);
  float testRMSE = bestModel.rmse(data.testSets);
  float valRMSE = bestModel.rmse(data.valSets);
  std::cout << "\nTrain sets RMSE: " << trainRMSE << std::endl;
  std::cout << "Test sets RMSE: " << testRMSE << std::endl;
  std::cout << "Val sets RMSE: " << valRMSE << std::endl;

  float trainRatingsRMSE   = bestModel.rmse(data.partTrainMat);
  float testRatingsRMSE    = bestModel.rmse(data.partTestMat);
  float valRatingsRMSE     = bestModel.rmse(data.partValMat);
  float notSetRMSE         = bestModel.rmseNotSets(data.allSets, data.ratMat);

  std::cout << "Train RMSE: " << trainRatingsRMSE << std::endl;
  std::cout << "Test RMSE: " << testRatingsRMSE << std::endl;
  std::cout << "Val RMSE: " << valRatingsRMSE << std::endl; 
  std::cout << "Test All Mat RMSE: " << notSetRMSE << std::endl;
  auto precisionNCall = bestModel.precisionNCall(data.allSets, data.ratMat,
      5, TOP_RAT_THRESH);
  std::cout << "Precision@5: " << precisionNCall.first << std::endl;
  std::cout << "OneCall@5: " << precisionNCall.second << std::endl;
  
  precisionNCall = bestModel.precisionNCall(data.allSets, data.ratMat,
      10, TOP_RAT_THRESH);
  std::cout << "Precision@10: " << precisionNCall.first << std::endl;
  std::cout << "OneCall@10: " << precisionNCall.second << std::endl;

  float corrOrdSetsTop = bestModel.fracCorrOrderedSets(data.valSets, 
      TOP_RAT_THRESH);
  std::cout << "Fraction top val correct ordered sets: " << corrOrdSetsTop 
    << std::endl;
  corrOrdSetsTop = bestModel.fracCorrOrderedSets(data.testSets, TOP_RAT_THRESH);
  std::cout << "Fraction top test correct ordered sets: " << corrOrdSetsTop 
    << std::endl;

  float corrOrdItemPairsTop = bestModel.corrOrderedItems(
                                          data.partValMat, TOP_RAT_THRESH);
  std::cout << "Fraction top val item pairs: " << corrOrdItemPairsTop << std::endl;
  
  corrOrdItemPairsTop = bestModel.corrOrderedItems(
                                          data.partTestMat, TOP_RAT_THRESH);
  std::cout << "Fraction top test item pairs: " << corrOrdItemPairsTop << std::endl;

  float corrOrdAllPairsTop = bestModel.matCorrOrderedRatingsWOSetsTop(data.allSets,
      data.ratMat, TOP_RAT_THRESH);
  std::cout << "Ordered top pairs excl all sets: " << corrOrdAllPairsTop 
    << std::endl;
  */
  return 0;
}


