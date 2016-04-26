#include <iostream>
#include <cstdlib>

#include "datastruct.h"
#include "ModelAverage.h"

Params parse_cmd_line(int argc, char* argv[]) {
  if (argc < 14) {
    std::cerr << "Not enough args" << std::endl;
    exit(1);
  }
  
  return Params(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]),
      std::atoi(argv[4]), std::atoi(argv[5]),
      std::atof(argv[6]), std::atof(argv[7]), std::atof(argv[8]), 
      std::atof(argv[9]),
      argv[10], argv[11], argv[12], argv[13], argv[14]);
}


int main(int argc, char *argv[]) {
  Params params = parse_cmd_line(argc, argv);
  params.display();
  Data data(params);

  //std::string opFName = std::string(params.prefix) + "_trainSet_temp";
  //writeSets(data.trainSets, opFName.c_str());

  ModelAverage modelAvg(params);
  Model bestModel(modelAvg);
  modelAvg.train(data, params, bestModel);

  std::cout << "Val RMSE: " << bestModel.rmse(data.valSets) << std::endl;
  std::cout << "Test RMSE: " << bestModel.rmse(data.testSets) << std::endl;

  return 0;
}



