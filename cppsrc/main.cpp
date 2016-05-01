#include <iostream>
#include <cstdlib>

#include "datastruct.h"
#include "ModelAverage.h"
#include "ModelAverageWCons.h"
#include "ModelMajority.h"
#include "ModelMajorityWCons.h"
#include "ModelBaseline.h"
#include "ModelAverageSigmoid.h"
#include "ModelAverageWBias.h"

Params parse_cmd_line(int argc, char* argv[]) {
  if (argc < 17) {
    std::cerr << "Not enough args" << std::endl;
    exit(1);
  }
  
  return Params(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]),
      std::atoi(argv[4]), std::atoi(argv[5]),
      std::atof(argv[6]), std::atof(argv[7]), std::atof(argv[8]), 
      std::atof(argv[9]), std::atof(argv[10]), std::atof(argv[11]), std::atof(argv[12]),
      argv[13], argv[14], argv[15], argv[16], argv[17]);
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

  std::cout << "Val RMSE: " << bestModel.rmse(data.valSets) << std::endl;
  std::cout << "Test RMSE: " << bestModel.rmse(data.testSets) << std::endl;

  //loadModelNRMSEs(data, params);

  return 0;
}



