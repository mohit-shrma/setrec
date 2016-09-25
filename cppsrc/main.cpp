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

#include "ModelFM.h"
#include "ModelMFWBias.h"
#include "ModelAverageHingeWBias.h"
#include "ModelAverageLogWBias.h"
#include "ModelAverageBPRWBias.h"
#include "ModelBPR.h"
#include "ModelBPRTop.h"
#include "ModelAverageBPRWBiasTop.h"
#include "ModelAverageGBiasWPart.h"
#include "ModelAverageBPRWPart.h"


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


void subSampleMats(gk_csr_t *mat, std::string prefix, int seed) {
  std::vector<double> pcs {0.01, 0.25, 0.5, 0.75};
  for (auto&& pc: pcs) {
    std::string fName = prefix + "_" + std::to_string(pc) + ".csr";
    writeSubSampledMat(mat, fName.c_str(), pc, seed);
  }
}


int main(int argc, char *argv[]) {
  Params params = parse_cmd_line(argc, argv);
  params.display();
  Data data(params);
  
  data.initRankMap(params.seed);

  std::cout << "Train users: " << data.trainUsers.size() << " Train items: " 
    << data.trainItems.size() << std::endl;

  //subSampleMats(data.partTrainMat, params.prefix, params.seed);

  ModelFM modelAvg(params);
  ModelFM bestModel(modelAvg);
  modelAvg.train(data, params, bestModel);
  //bestModel.save(params.prefix);
  //bestModel.load(params.prefix);

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
  
  std::vector<int> Ns {1, 5, 10, 25, 50, 100};
  std::vector<float> precisions;
  std::vector<float> oneCalls;

  auto precisionNCalls = bestModel.precisionNCall(data.allSets, data.ratMat, Ns,
      TOP_RAT_THRESH);
  precisions = precisionNCalls.first; 
  oneCalls   = precisionNCalls.second;

  std::cout << "Precision: ";
  for (auto&& prec : precisions) {
    std::cout << prec << " ";
  }
  std::cout << std::endl;
  
  std::cout << "OneCall: ";
  for (auto&& oneCall : oneCalls) {
    std::cout << oneCall << " ";
  }
  std::cout << std::endl;
  
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
  
  return 0;
}


