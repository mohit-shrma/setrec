#include "setrec.h"


void parse_cmd_line(int argc, char **argv) {
  
  Params *params;
  Data *data = NULL;
  float *baseValTest, *simValTest, *tempValTest;
  int i;

  params = (Params *) malloc(sizeof(Params));
  memset(params, 0, sizeof(Params));
 
  tempValTest = (float*) malloc(sizeof(float)*2);
  memset(tempValTest, 0, sizeof(float)*2);
  
  baseValTest = (float*) malloc(sizeof(float)*2);
  memset(baseValTest, 0, sizeof(float)*2);

  simValTest = (float*) malloc(sizeof(float)*2);
  memset(simValTest, 0, sizeof(float)*2);

  if (argc < 11) {
    printf("\n Error: need args");
    exit(0);
  } else {
    params->user_set_file   = argv[1];
    params->nUsers          = atoi(argv[2]);
    params->nItems          = atoi(argv[3]);
    params->facDim          = atoi(argv[4]);
    params->regU            = atof(argv[5]);
    params->regI            = atof(argv[6]);
    params->learnRate       = atof(argv[7]);
    params->useSim          = atoi(argv[8]);
    params->maxIter         = atoi(argv[9]);
    params->seed            = atoi(argv[10]);
  }

  //initialize random seed
  srand(params->seed);

  //load data
  data = (Data *) malloc(sizeof(Data));
  Data_init(data, params->nUsers, params->nItems);
  
  //printf("\nloading data...");
  loadData(data, params);

  //printf("\ndisplaying data...");
  //writeData(data);

  for (i = 0; i <3; i++) {
    
    //run baseline
    memset(tempValTest, 0, sizeof(float)*2);
    modelBase(data, params, tempValTest);  
    baseValTest[0] += tempValTest[0];
    baseValTest[1] += tempValTest[1];

    //learn model
    memset(tempValTest, 0, sizeof(float)*2);
    modelSim(data, params, tempValTest);
    simValTest[0] += tempValTest[0];
    simValTest[1] += tempValTest[1];

    //reset test and val for next iter
    srand(params->seed + (i+1));
    Data_reset(data, params->nUsers, params->nItems);
  }

  printf("\navg baseline validation: %f test: %f", baseValTest[0]/i, 
      baseValTest[1]/i);
  printf("\navg sim validation: %f test: %f", simValTest[0]/i, simValTest[1]/i);

  if (baseValTest[1]/i > simValTest[1]/i) {
    printf("\n* %f %f", baseValTest[1]/i, simValTest[1]/i);
  }


  Data_free(data);  
  free(params);
  free(baseValTest);
  free(simValTest);
  free(tempValTest);
}


int main(int argc, char *argv[]) {
  parse_cmd_line(argc, argv);
  return 0;
}

