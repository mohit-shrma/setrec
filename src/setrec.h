#ifndef _SETREC_H
#define _SETREC_H

typedef struct {

  char *user_set_file;
  char *artist_set_file;

} Params;


void parse_cmd_line(int argc, char **argv);

#endif
