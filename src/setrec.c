#include "setrec.h"


void parse_cmd_line(int argc, char **argv) {
  
  Params *params;
  Data *data = NULL;
  params = (Params *) malloc(sizeof(Params));
  memset(params, 0, sizeof(Params));
 
  if (argc < 2) {
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
  }

  //load data
  data = (Data *) malloc(sizeof(Data));
  Data_init(data, params->nUsers, params->nItems);
  
  //printf("\nloading data...");
  loadData(data, params);

  //printf("\ndisplaying data...");
  writeData(data);

  //run model
  
  Data_free(data);  
  free(params);
}


int main(int argc, char *argv[]) {
  parse_cmd_line(argc, argv);
  return 0;
}

