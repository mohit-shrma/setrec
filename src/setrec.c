#include "setrec.h"


void parse_cmd_line(int argc, char **argv) {
  
  Params *params;
  Data *data = NULL;
  ValTestRMSE *valTest;
  int i;

  params = (Params *) malloc(sizeof(Params));
  memset(params, 0, sizeof(Params));
  
  valTest = (ValTestRMSE *) malloc(sizeof(ValTestRMSE));
  memset(valTest, 0, sizeof(ValTestRMSE));

  if (argc < 20) {
    printf("\n Error: need args");
    exit(0);
  } else {
    params->user_set_file   = argv[1];
    params->nUsers          = atoi(argv[2]);
    params->nItems          = atoi(argv[3]);
    params->facDim          = atoi(argv[4]);
    params->regU            = atof(argv[5]);
    params->regI            = atof(argv[6]);
    params->constrainWt     = atof(argv[7]);
    params->learnRate       = atof(argv[8]);
    params->useSim          = atoi(argv[9]);
    params->maxIter         = atoi(argv[10]);
    params->seed            = atoi(argv[11]);

    params->train_mat_file  = argv[12];
    params->test_mat_file   = argv[13];
    params->val_mat_file    = argv[14];
    
    params->ext_setSim_file = argv[15];
    
    params->rhoRMS          = atof(argv[16]);
    params->epsRMS          = atof(argv[17]);
    
    params->uFacFileName    = NULL;//argv[18];
    params->iFacFileName    = NULL;//argv[19];
    params->uMidPFName      = argv[20];
  }

  //initialize random seed
  srand(params->seed);

  //load data
  data = (Data *) malloc(sizeof(Data));
  Data_init(data, params->nUsers, params->nItems);

  printf("\nloading data...");
  fflush(stdout);
  loadData(data, params);
  ItemSets_init(data->itemSets, data->userSets, params->nUsers, 
      params->nItems);
  
  printf("\nFinished loading data..");
  fflush(stdout);

  //load external similarities
  //loadSims(data, params);
  //writeSims(data);

  //printf("\ndisplaying data...");
  //writeData(data);

  //learn model
  //modelLogisticWUm(data, params, valTest);
  modelHingeSqrWUm(data, params, valTest);  

  //reset test and val for next iter
  //srand(params->seed + (i+1));
  //Data_reset(data, params->nUsers, params->nItems);

  printf("\nRE: %f %f %f %d %f %f"
          " %f %f %f"
          " %f %f %f"
          " %f %f %.5e", 
      params->regU, params->regI, params->constrainWt, params->facDim, params->learnRate, params->rhoRMS, 
      valTest->trainItemsRMSE, valTest->trainSetRMSE, valTest->testItemsRMSE, 
      valTest->testSetRMSE, valTest->testSpearman, valTest->valItemsRMSE,
      valTest->valSetRMSE, valTest->valSpearman, valTest->setObj);

  Data_free(data);  
  free(params);
  free(valTest);
}


int main(int argc, char *argv[]) {
  parse_cmd_line(argc, argv);
  return 0;
}


