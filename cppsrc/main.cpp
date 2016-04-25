#include <iostream>
#include <cstdlib>
#include "datastruct.h"

Params parse_cmd_line(int argc, char* argv[]) {
  if (argc < 13) {
    std::cerr << "Not enough args" << std::endl;
    exit(1);
  }
  
  return Params(std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3]),
      std::atoi(argv[4]), std::atoi(argv[5]),
      std::atof(argv[6]), std::atof(argv[7]), std::atof(argv[8]),
      argv[9], argv[10], argv[11], argv[12], argv[13]);
}


int main(int argc, char *argv[]) {
  Params params = parse_cmd_line(argc, argv);
  params.display();
  Data data(params);
  return 0;
}
