#include "setrec.h"


void parse_cmd_line(int argc, char **argv) {
  
  Params *params;
  params = (Params *) malloc(sizeof(Params));
  memset(params, 0, sizeof(Params));
 
  if (argc < 2) {
    printf("\n Error: need args");
    exit(0);
  } else {
    params->user_set_file   = argv[1];
    params->nUsers          = atoi(argv[2]);
    params->nItems          = atoi(argv[3]);
  }

  //load data
  
  //run model
}



int main(int argc, char *argv[]) {
  parse_cmd_line(argc, argv);
  return 0;
}

