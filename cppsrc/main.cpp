#include <iostream>
#include <cstdlib>

#include "datastruct.h"
#include "ModelAverage.h"
#include "ModelAverageWBias.h"
#include "ModelAverageWPart.h"
#include "ModelBaseline.h"
#include "ModelAverageBiasesOnly.h"
#include "ModelAverageWGBias.h"

#include "ModelMaxMin.h"
#include "ModelWeightedVariance.h"
#include "ModelWtAverageAllRange.h"
#include "ModelWtAverage.h"
#include "ModelAverageSigmoidWBias.h"
#include "ModelFMUWt.h"
#include "ModelFMUWtBPR.h"
#include "ModelMFWBias.h"
#include "ModelAverageBPRWBias.h"
#include "ModelBPR.h"
#include "ModelBPRTop.h"
#include "ModelAverageBPRWBiasTop.h"
#include "ModelAverageGBiasWPart.h"
#include "ModelAverageBPRWPart.h"
#include "ModelWeightedVarianceWBias.h"
#include "ModelWtAverageAllRangeWBias.h"


Params parse_cmd_line(int argc, char* argv[]) {
  if (argc < 23) {
    std::cerr << "Not enough args" << std::endl;
    exit(1);
  }
  
  return Params(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]),
      std::atoi(argv[4]), std::atoi(argv[5]),
      std::atof(argv[6]), std::atof(argv[7]), std::atof(argv[8]),
      std::atof(argv[9]), std::atof(argv[10]),
      std::atof(argv[11]), std::atof(argv[12]), std::atoi(argv[13]), std::atof(argv[14]),
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
  //std::vector<double> pcs {0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75};
  std::vector<double> pcs { 0.5};
  for (auto&& pc: pcs) {
    std::string fName = prefix + "_" + std::to_string(pc) + ".csr";
    std::cout << "Writing... " << fName << std::endl;
    writeSubSampledMat(mat, fName.c_str(), pc, seed);
  }
}


void writeTrainTestValMat(gk_csr_t *mat,  const char* trainFileName, 
    const char* testFileName, const char *valFileName, float testPc, 
    float valPc, int seed) {
  int k, i;
  int nnz = getNNZ(mat);
  int nTest = testPc * nnz;
  int nVal = valPc * nnz;
  int* color = (int*) malloc(sizeof(int)*nnz);
  memset(color, 0, sizeof(int)*nnz);
 
  //initialize uniform random engine
  std::mt19937 mt(seed);
  //nnz dist
  std::uniform_int_distribution<int> nnzDist(0, nnz-1);

  for (i = 0; i < nTest; i++) {
    k = nnzDist(mt);
    color[k] = 1;
  }
  
  i = 0;
  while (i < nVal) {
    k = nnzDist(mt);
    if (!color[k]) {
      color[k] = 2;
      i++;
    }
  }


  //split the matrix based on color
  gk_csr_t** mats = gk_csr_Split(mat, color);
  
  //save first matrix as train
  gk_csr_Write(mats[0], (char*) trainFileName, GK_CSR_FMT_CSR, 1, 0);

  //save second matrix as test
  gk_csr_Write(mats[1], (char*) testFileName, GK_CSR_FMT_CSR, 1, 0);

  //save third matrix as val
  gk_csr_Write(mats[2], (char*) valFileName, GK_CSR_FMT_CSR, 1, 0);
  
  free(color);
  gk_csr_Free(&mats[0]);
  gk_csr_Free(&mats[1]);
  gk_csr_Free(&mats[2]);
  //TODO: free mats
  //gk_csr_Free(&mats);
}


int main(int argc, char *argv[]) {
  
  //used below to generate 
  
  //gk_csr_t *mat = gk_csr_Read(argv[1], GK_CSR_FMT_CSR, 1, 0);
  //subSampleMats(mat, argv[2], atoi(argv[3]));
  //return 0;
  

  //partition the given matrix into train test val
  /*
  gk_csr_t *mat = gk_csr_Read(argv[1], GK_CSR_FMT_CSR, 1, 0);
  writeTrainTestValMat(mat, argv[2], argv[3], argv[4], 0.1, 0.1, atoi(argv[5]));  
  return 0;
  */ 
  
  Params params = parse_cmd_line(argc, argv);
  params.display();
  Data data(params);
  
  data.initRankMap(params.seed);
  //data.computeSetsEntropy();
  //data.writeTrainSetsEntropy();
  //data.scaleSetsTo01(5.0); 

  std::cout << "Train users: " << data.trainUsers.size() << " Train items: " 
    << data.trainItems.size() << std::endl;

  ModelWtAverageAllRange modelAvg(params);
  ModelWtAverageAllRange bestModel(modelAvg);
  modelAvg.trainQPSmooth(data, params, bestModel);
  /*
  if (argc > 23) {
    std::cout << "UFac norm: " << bestModel.U.norm() << " iFac norm: " << bestModel.V.norm() << std::endl;
    std::cout << "Reading... uFac: " << argv[23] << std::endl;
    readEigenMat(argv[23], bestModel.U, bestModel.nUsers, bestModel.facDim);
    std::cout << "Reading... iFac: " << argv[24] << std::endl;
    readEigenMat(argv[24], bestModel.V, bestModel.nItems, bestModel.facDim);
    //std::cout << "Reading... UWts: " << argv[25] << std::endl;
    //readEigenMat(argv[25], bestModel.UWts, bestModel.nUsers, bestModel.nWts);
    std::cout << "Reading.. uDivWts: " << argv[25] << std::endl;
    readEigenVec(argv[25], bestModel.uDivWt, bestModel.nUsers);
    std::cout << "UFac norm: " << bestModel.U.norm() << " iFac norm: " << bestModel.V.norm() << std::endl;
  }
  */
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
  //float notSetRMSE       = bestModel.rmseNotSets(data.allSets, data.ratMat);
  float notSetRMSE         = bestModel.rmseNotSets(data.allSets, data.ratMat, 
                                                   data.partTrainMat);

  std::cout << "Train RMSE: " << trainRatingsRMSE << std::endl;
  std::cout << "Test RMSE: " << testRatingsRMSE << std::endl;
  std::cout << "Val RMSE: " << valRatingsRMSE << std::endl; 
  std::cout << "Test All Mat RMSE: " << notSetRMSE << std::endl;
  
  std::vector<int> Ns {1, 5, 10, 25, 50, 100};
  std::vector<float> precisions;
  std::vector<float> oneCalls;
  
  /*
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
  
  //float corrOrdSetsTop = bestModel.fracCorrOrderedSets(data.valSets, 
  //    TOP_RAT_THRESH);
  float corrOrdSetsTop = bestModel.fracCorrOrderedSets(data.valSets);
  std::cout << "Fraction top val correct ordered sets: " << corrOrdSetsTop 
    << std::endl;
  //corrOrdSetsTop = bestModel.fracCorrOrderedSets(data.testSets, TOP_RAT_THRESH);
  corrOrdSetsTop = bestModel.fracCorrOrderedSets(data.testSets);
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
  params.display();  
  
  return 0;
}


