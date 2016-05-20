#include <iostream>
#include <cstdlib>

#include "datastruct.h"
#include "ModelAverage.h"
#include "ModelAverageWBias.h"
#include "ModelAverageWSetBias.h"
#include "ModelBaseline.h"
#include "ModelMajority.h"
#include "ModelMajorityWCons.h"
#include "ModelAverageBPR.h"
#include "ModelAverageHinge.h"
#include "ModelAverageWPart.h"
#include "ModelBaseline.h"
#include "ModelAverageBiasesOnly.h"
#include "ModelAverageWGBias.h"
#include "ModelItemAverage.h"
#include "ModelAverageSetBiasWPart.h"
#include "ModelMFWBias.h"

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

  std::cout << "Train users: " << data.trainUsers.size() << " Train items: " 
    << data.trainItems.size() << std::endl;
  
  //std::string opFName = std::string(params.prefix) + "_trainSet_temp";
  //writeSets(data.trainSets, opFName.c_str());

  ModelAverageWBias modelAvg(params);
  //ModelAverageSigmoid modelAvgSigmoid(params);
  //data.scaleSetsTo01(5.0);
  //ModelBaseline modelBase(params);
  //ModelMajority modelMaj(params);
  //ModelMajorityWCons modelMaj(params);
  
  ModelAverageWBias bestModel(modelAvg);
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
  
  float trainRMSE = bestModel.rmse(data.trainSets);
  float testRMSE = bestModel.rmse(data.testSets);
  float valRMSE = bestModel.rmse(data.valSets);
  std::cout << "Train sets RMSE: " << trainRMSE << std::endl;
  std::cout << "Test sets RMSE: " << testRMSE << std::endl;
  std::cout << "Val sets RMSE: " << valRMSE << std::endl;
  
  //float trainRatingsRMSE = bestModel.rmse(data.trainSets, data.ratMat);
  //float testRatingsRMSE  = bestModel.rmse(data.testSets, data.ratMat);
  //float valRatingsRMSE   = bestModel.rmse(data.valSets, data.ratMat);
  float trainRatingsRMSE   = bestModel.rmse(data.partTrainMat);
  float testRatingsRMSE    = bestModel.rmse(data.partTestMat);
  float valRatingsRMSE     = bestModel.rmse(data.partValMat);
  std::cout << "Train RMSE: " << trainRatingsRMSE << std::endl;
  std::cout << "Test RMSE: " << testRatingsRMSE << std::endl;
  std::cout << "Val RMSE: " << valRatingsRMSE << std::endl; 

  //std::cout << "Under sets RMSE: " << bestModel.rmse(undSets) << std::endl;
  //std::cout << "Over sets RMSE: " << bestModel.rmse(ovrSets) << std::endl;

  //float recN = bestModel.recallTopN(data.ratMat, data.trainSets, 10);
  //float spN = bestModel.spearmanRankN(data.ratMat, data.trainSets, 10);

  /*
  auto itemFreq = getItemFreq(data.trainSets);
  
  auto trainItemsRMSE = bestModel.itemRMSE(data.trainSets, data.ratMat);
  writeItemRMSEFreq(itemFreq, trainItemsRMSE, "trainItemsRMSE.txt");

  auto testItemsRMSE = bestModel.itemRMSE(data.testSets, data.ratMat);
  writeItemRMSEFreq(itemFreq, testItemsRMSE, "testItemsRMSE.txt");
  
  auto valItemsRMSE = bestModel.itemRMSE(data.valSets, data.ratMat);
  writeItemRMSEFreq(itemFreq, valItemsRMSE, "valItemsRMSE.txt");
  */

  //std::cout << "Inversion count: " << bestModel.inversionCount(data.partTestMat, 
  //    data.trainSets, 10) << std::endl;
  std::cout << "Random inversion count: " 
    << bestModel.invertRandPairCount(data.partTestMat, data.trainSets, 
        params.seed) <<std::endl;
  /*
  std::vector<int> invalItems = readVector(
      "mfbias_854X12548_5_0.010000_0.010000_0.001000_invalItems.txt");
  std::unordered_set<int> invalSet(invalItems.begin(), invalItems.end());
  std::cout << "inval RMSE: " << bestModel.rmse(data.partTestMat, invalSet) << std::endl;
  */
  /* 
  std::cout << "\nRE: " <<  params.facDim << " " << params.uReg << " " 
    << params.iReg << " " << params.learnRate << " " << trainRMSE << " " 
    << testRMSE << " " << valRMSE << " " << trainRatingsRMSE << " " 
    << testRatingsRMSE << " " << valRatingsRMSE
    << " " << recN <<  " " << spN << std::endl;
  */

  return 0;
}


