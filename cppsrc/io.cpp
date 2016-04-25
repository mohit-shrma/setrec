#include "io.h"

std::vector<UserSets> readSets(const char* fileName) {
  
  std::vector<UserSets> uSets;
  size_t pos;
  std::string line, token;
  std::ifstream inFile(fileName);
  std::string delimiter = " ";

  int user, numSets, nUItems;
  std::vector<int> uItems;
  std::vector<int> setItems;
  
  float rating;
  int setSz;
  std::vector<int> itemSet;

  std::vector<std::vector<int>> itemSets;
  std::vector<float> setScores;

  if (inFile.is_open()) { 
  
    while (getline(inFile, line)) {
      //split the header line
      //get user
      pos = line.find(delimiter);
      if (pos != std::string::npos) {
        token = line.substr(0, pos);
        user = std::stoi(token);
        line.erase(0, pos + delimiter.length());
      }

      //get numsets
      pos = line.find(delimiter);
      if (pos != std::string::npos) {
        token = line.substr(0, pos);
        numSets = std::stoi(token); 
        line.erase(0, pos + delimiter.length());
      }
      
      //get nItems
      pos = line.find(delimiter);
      if (pos != std::string::npos) {
        token = line.substr(0, pos);
        nUItems = std::stoi(token);  
        line.erase(0, pos + delimiter.length());
      }
      
      //get items
      uItems.clear();
      while((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        uItems.push_back(std::stoi(token)); 
        line.erase(0, pos + delimiter.length());
      }
      if (line.length > 0) {
        uItems.push_back(std::stoi(line));
      }
      
      if (nUItems != uItems.size()) {
        std::cerr << "No. of set items dont match." << std::endl;
        exit(1);
      }

      //read sets for the user
      itemSets.clear();
      setScores.clear();
      int i = 0;
      while (getline(inFile, line) && i < numSets) {
        
        //rating
        pos = line.find(delimiter);
        if (pos != std::string::npos) {
          token = line.substr(0, pos);
          rating = std::stof(token);  
          line.erase(0, pos + delimiter.length());
        }
        setScores.push_back(rating);

        //nItems in set
        pos = line.find(delimiter);
        if (pos != std::string::npos) {
          token = line.substr(0, pos);
          setSz = std::stoi(token);  
          line.erase(0, pos + delimiter.length());
        }

        //items in the set
        itemSet.clear() 
        while((pos = line.find(delimiter)) != std::string::npos) {
          token = line.substr(0, pos);
          itemSet.push_back(std::stoi(token)); 
          line.erase(0, pos + delimiter.length());
        }
        if (line.length > 0) {
          itemSet.push_back(std::stoi(line));
        }
        
        if (setSz != itemSet.size()) {
          //size of the set didnt match the specified number
          std::cerr << "No. of items in the set dont match." << std::endl;
        }
        itemSets.push_back(itemSet);

      }
      
      uSets.push_back(UserSets(user, itemSets, setScores));

    }
  
  }
  
  return uSets;
}


void writeSets(std::vector<UserSets> uSets, const char* opFName) {
  //TODO:

  
}


