#ifndef _SETREC_H
#define _SETREC_H

#include <stdio.h>
#include <stdlib.h>

typedef struct {

  char *user_set_file;
  int nUsers;
  int nItems;
} Params;


void parse_cmd_line(int argc, char **argv);

#endif
