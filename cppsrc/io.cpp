#include "io.h"

std::vector<UserSets> readSets(const char* fileName) {
  
  std::vector<UserSets> uSets;
  size_t pos;
  std::string line, token;
  std::ifstream inFile(fileName);
  std::string delimiter = " ";

  int user = -1, numSets = -1, nUItems = -1, nSets = 0;
  std::vector<int> uItems;
  std::vector<int> setItems;
  int maxUser = -1, maxItem = -1; 
  float rating = -1;
  int setSz = -1;
  std::vector<int> itemSet;
  std::unordered_set<int> items;
  std::vector<std::pair<std::vector<int>,float>> itemSets;

  if (inFile.is_open()) {
    std::cout << "Reading..." << fileName << std::endl;
    while (getline(inFile, line)) {
      //split the header line
      //get user
      pos = line.find(delimiter);
      if (pos != std::string::npos) {
        token = line.substr(0, pos);
        user = std::stoi(token);
        line.erase(0, pos + delimiter.length());
      }

      if (user > maxUser) {
        maxUser = user;
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
      if (line.length() > 0) {
        uItems.push_back(std::stoi(line));
      }
      
      for (auto&& item: uItems) {
        if (maxItem < item) {
          maxItem = item;
        }
      }

      if (nUItems != (int)uItems.size()) {
        std::cerr << "No. of set items dont match." << std::endl;
        exit(1);
      }

      //read sets for the user
      itemSets.clear();
      int i = 0;
      while (i < numSets && getline(inFile, line)) {
        //rating
        pos = line.find(delimiter);
        if (pos != std::string::npos) {
          token = line.substr(0, pos);
          rating = std::stof(token);  
          line.erase(0, pos + delimiter.length());
        }

        //nItems in set
        pos = line.find(delimiter);
        if (pos != std::string::npos) {
          token = line.substr(0, pos);
          setSz = std::stoi(token);  
          line.erase(0, pos + delimiter.length());
        }

        //items in the set
        itemSet.clear(); 
        while((pos = line.find(delimiter)) != std::string::npos) {
          token = line.substr(0, pos);
          itemSet.push_back(std::stoi(token));
          items.insert(std::stoi(token));
          line.erase(0, pos + delimiter.length());
        }
        if (line.length() > 0) {
          itemSet.push_back(std::stoi(line));
          items.insert(std::stoi(line));
        }
        
        if (setSz != (int)itemSet.size()) {
          //size of the set didnt match the specified number
          std::cerr << "No. of items in the set dont match: " << setSz 
            << " " << itemSet.size() << " " << user << " " << rating 
            << " " << setSz << std::endl;
          exit(1);
        }
        itemSets.push_back(std::make_pair(itemSet, rating));
        
        i++;
        nSets++;
      }
      
      uSets.push_back(UserSets(user, itemSets));
      
      if (itemSets.size() == 0) {
        std::cerr << "size of itemsets is 0 " << user << std::endl;
      }

    }
    inFile.close(); 
  } else {
    std::cerr << "Can't open file: " << fileName << std::endl;
  } 

  std::cout << "No. of UserSets: " << uSets.size() << std::endl;
  std::cout << "No. of sets: " << nSets << std::endl;
  std::cout << "No. of items: " << items.size() << std::endl;
  std::cout << "max user: " << maxUser << std::endl;  
  std::cout << "max item: " << maxItem << std::endl;  

  return uSets;
}


void writeSets(std::vector<UserSets> uSets, const char* opFName) {
  std::ofstream opFile(opFName);
  if (opFile.is_open()) {
    
    for (auto&& uSet: uSets) {
      //write out user , number of sets, items
      opFile << uSet.user << " " << uSet.itemSets.size() << " " 
        << uSet.items.size() << " ";
      for (auto&& item: uSet.items) {
        opFile << item << " ";
      }
      opFile << std::endl;
      
      //write out set ratings and set details
      for (size_t i = 0; i < uSet.itemSets.size(); i++) {
        opFile << uSet.itemSets[i].second << " " << uSet.itemSets[i].first.size() << " ";
        for (auto&& item: uSet.itemSets[i].first) {
          opFile << item << " ";
        }
        opFile << std::endl;
      }
    }
   
    opFile.close();
  } else {
    std::cerr << "Can't open file: " << opFName << std::endl;
  } 
}

//TODO:verify
void readEigenMat(const char* fileName, Eigen::MatrixXf& mat, int nrows, 
    int ncols) {
  
  size_t pos;
  std::string line, token;
  std::ifstream ipFile(fileName);
  std::string delimiter = " ";
  int rowInd = 0, colInd = 0;

  if (ipFile.is_open()) {
    std::cout << "Reading... " << fileName << std::endl;  
    while (getline(ipFile, line)) {
      int colInd = 0;
      while((pos = line.find(delimiter)) != std::string::npos) {
        token = line.substr(0, pos);
        mat(rowInd, colInd) = std::stof(token);
        colInd++;
        line.erase(0, pos + delimiter.length());
      }
      if (line.length() > 0) {
        mat(rowInd, colInd) = std::stof(line);
      }
      rowInd++;
    } 
    ipFile.close();
  }
  
  std::cout << "Read: nrows" << rowInd << " ncols: " << colInd << std::endl;
}


