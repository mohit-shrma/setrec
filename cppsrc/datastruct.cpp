#include "datastruct.h"


  Data::Data(const Params& params) {
      
    nUsers = params.nUsers;
    nItems = params.nItems;
    prefix = params.prefix;

    if (NULL != params.ratMatFile) {
      std::cout << "\nReading rating matrix 0 indexed..." << params.ratMatFile;
      ratMat = gk_csr_Read(params.ratMatFile, GK_CSR_FMT_CSR, 1, 0);
      gk_csr_CreateIndex(ratMat, GK_CSR_COL);
    }
    
    if (NULL != params.partTrainMatFile) {
      std::cout << "\nReading partial rating matrix 0 indexed..." 
        << params.partTrainMatFile << std::endl;
      partTrainMat = gk_csr_Read(params.partTrainMatFile, GK_CSR_FMT_CSR, 1, 0);
      gk_csr_CreateIndex(partTrainMat, GK_CSR_COL);
      std::cout << "nUsers: " << partTrainMat->nrows 
        << " nItems: " << partTrainMat->ncols << std::endl;
    }
    
    if (NULL != params.partTestMatFile) {
      std::cout << "\nReading partial rating matrix 0 indexed..." 
        << params.partTestMatFile << std::endl;
      partTestMat = gk_csr_Read(params.partTestMatFile, GK_CSR_FMT_CSR, 1, 0);
      std::cout << "nUsers: " << partTestMat->nrows 
        << " nItems: " << partTestMat->ncols << std::endl;
    } 
    
    if (NULL != params.partValMatFile) {
      std::cout << "\nReading partial rating matrix 0 indexed..." 
        << params.partValMatFile << std::endl;
      partValMat = gk_csr_Read(params.partValMatFile, GK_CSR_FMT_CSR, 1, 0);
      std::cout << "nUsers: " << partValMat->nrows 
        << " nItems: " << partValMat->ncols << std::endl;
    }
    
    std::cout << "\n";
    
    if (NULL != params.trainSetFile) {
      trainSets  = readSets(params.trainSetFile);    
      
      //compute entropy
      for (auto&& uSet: trainSets) {
        uSet.computeEntropy(ratMat);
        //remove high entropy
        //uSet.removeRandom(0.5, params.seed);
        //uSet.removeHighEntropy(0.5);
        //uSet.removeLowEntropy(0.5);
      }

      //remove over-under rated sets
      //removeOverUnderRatedSets(trainSets, ratMat);
      std::cout << "trainSets and partTrainMat differ: " << std::endl; 
      checkIfSetsMatDiffer(trainSets, partTrainMat);

      std::cout << "trainSets and partTestMat overlap: " << std::endl; 
      auto invalU1 = checkIfSetsMatOverlap(trainSets, partTestMat);

      std::cout << "trainSets and partValMat overlap: " << std::endl; 
      auto invalU2 = checkIfSetsMatOverlap(trainSets, partValMat);
      
      //removeSetsWInvalUsers(trainSets, invalU1);
      //removeSetsWInvalUsers(trainSets, invalU2);

      std::cout << "No. of train users: " << trainSets.size() << std::endl;
      nTrainSets = 0;
      for (auto&& uSet: trainSets) {
        nTrainSets += uSet.itemSets.size();
      }
      auto trainUserItems = getUserItems(trainSets);
      trainUsers = trainUserItems.first;
      trainItems = trainUserItems.second;
      std::cout << "nTrainSets: " << nTrainSets << std::endl;
      std::cout << "nTrainUsers: " << trainUsers.size() << std::endl;
      std::cout << "nTrainItems: " << trainItems.size() << std::endl;
    }

    std::cout << "\n";

    if (NULL != params.testSetFile) {
      testSets = readSets(params.testSetFile);
      nTestSets = 0;
      //remove test sets which contain items not present in train
      removeSetsWOValItems(testSets, trainItems);
      //remove over-under rated sets
      //removeOverUnderRatedSets(testSets, ratMat);

      for (auto&& uSet: testSets) {
        nTestSets += uSet.itemSets.size();
      }
      std::cout << "No. of test users: " << testSets.size() << std::endl;
      std::cout << "nTestSets: " << nTestSets << std::endl;
    }

    std::cout << "\n";
    
    if (NULL != params.valSetFile) {
      valSets = readSets(params.valSetFile);
      nValSets = 0;
      
      //remove val sets which contain items not present in train
      removeSetsWOValItems(valSets, trainItems);
      //remove over-under rated sets
      //removeOverUnderRatedSets(valSets, ratMat);

      for (auto&& uSet: valSets) {
        nValSets += uSet.itemSets.size();
      }
      std::cout << "No. of val users: " << valSets.size() << std::endl;
      std::cout << "nValSets: " << nValSets << std::endl;
    }
    
    std::cout << "\n";
    
    //merge test val sets for common users
    testValMergeSets = merge(testSets, valSets);
    
    //merge train with test val sets
    allSets = merge(trainSets, testValMergeSets);

    testValRatings = getUIRatings(partTestMat, partValMat, nUsers);
  }


    void Data::scaleSetsToSigm(std::vector<UserSets> uSets, 
        std::map<int, float> uMidps, float g_k) {
      for (auto&& uSet: uSets) {
        int user = uSet.user;
        if (uMidps.find(user) != uMidps.end()) {
          //found user midp
          uSet.scaleToSigm(uMidps[user], g_k);
        }
      }
    }


    void Data::scaleSetsTo01(float maxRat) {
      for (auto&& uSet: trainSets) {
        uSet.scaleTo01(maxRat);
      }
      for (auto&& uSet: testSets) {
        uSet.scaleTo01(maxRat);
      }
      for (auto&& uSet: valSets) {
        uSet.scaleTo01(maxRat);
      }
    }
   

    void Data::computeSetsEntropy() {
      for (auto&& uSet: trainSets) {
        uSet.computeEntropy(ratMat);
      }
      for (auto&& uSet: testSets) {
        uSet.computeEntropy(ratMat);
      }
      for (auto&& uSet: valSets) {
        uSet.computeEntropy(ratMat);
      }
    }


    void Data::removeInvalUI() {
      //set of valid items = trainItems - invalItems
      std::unordered_set<int> valItems;
      for (auto item: trainItems) {
        if (invalItems.find(item) == invalItems.end()) {
          //train item is valid
          valItems.insert(item);
        }
      }

      //set of valid users = trainUsers - invalUsers
      std::unordered_set<int> valUsers;
      for (auto user: trainUsers) {
        if (invalUsers.find(user) == invalUsers.end()) {
          //train user is valid
          valUsers.insert(user);
        }
      }
      
      removeInvalUIFrmSets(trainSets, valUsers, valItems);
      //update train users and items
      userItemsFrmSets(trainSets, trainUsers, trainItems);
      nTrainSets = trainSets.size();

      removeInvalUIFrmSets(testSets, valUsers, valItems);
      nTestSets = testSets.size();

      removeInvalUIFrmSets(valSets, valUsers, valItems);
      nValSets = valSets.size();
    }


    void Data::initRankMap(int seed) {
      std::vector<std::pair<int, float>> itemActRatings;
      for (auto&& uSet: trainSets) {
        int user = uSet.user;
        auto setItems = uSet.items;
        itemActRatings.clear();

        for (int ii = ratMat->rowptr[user]; ii < ratMat->rowptr[user+1]; ii++) {
          int item = ratMat->rowind[ii];
          if (setItems.find(item) == setItems.end() 
              && trainItems.find(item) != trainItems.end()) {
            //not found in user sets, but exist in train items
            itemActRatings.push_back(std::make_pair(item, ratMat->rowval[ii]));
          }
        }
        
        std::map<int, float> valMap;
        std::map<int, float> testMap;

        //shuffle before partitioning
        //initialize random engine
        std::mt19937 mt(seed);
        std::shuffle(itemActRatings.begin(), itemActRatings.end(), mt);

        if (itemActRatings.size() >= 4) {
          for (auto it = itemActRatings.begin(); 
              it != itemActRatings.begin() + itemActRatings.size()/2; it++) {
            valMap[(*it).first] = (*it).second;
          }
          for (auto it = itemActRatings.begin() + itemActRatings.size()/2; 
              it != itemActRatings.end(); it++) {
            testMap[(*it).first] = (*it).second;
          }
          valURatings[user]  = valMap;
          testURatings[user] = testMap;
        }

        std::sort(itemActRatings.begin(), itemActRatings.end(), descComp);
        auto invertedPairs = getInvertItemPairs(itemActRatings, 10, mt); 
        for (auto&& kv: invertedPairs) {
          auto item = kv.first;
          auto loItems = kv.second;
          for (auto&& loItem: loItems) {
            allTriplets.push_back(std::make_tuple(user, item, loItem));    
          }
        }

        //get top-2 elements in beginning
        std::nth_element(itemActRatings.begin(), itemActRatings.begin()+(2-1),
            itemActRatings.end(), descComp);

        if (!(itemActRatings[0].second > 3 && itemActRatings[1].second > 3)) {
          continue;
        }

        valUItems[user] = itemActRatings[0].first;
        testUItems[user] = itemActRatings[1].first;

        std::unordered_set<int> uIgnoreItems;
        for (auto it = itemActRatings.begin()+2; it != itemActRatings.end(); 
            it++) {
          if ( (*it).second > 3) {
            uIgnoreItems.insert((*it).first);
          }
        }
        ignoreUItems[user] = uIgnoreItems;
      }
    }


    void Data::writeTrainSetsEntropy() {
      std::string opFName = "uSetsEntropy.txt";
      std::ofstream opFile(opFName);
      for (auto&& uSet: trainSets) {
        uSet.computeEntropy(ratMat);
        
        int user = uSet.user;
        
        std::map<int, float> itemRatings;
        for (int ii = ratMat->rowptr[user]; 
            ii < ratMat->rowptr[user+1]; ii++) {
          int item = ratMat->rowind[ii];
          float rating = ratMat->rowval[ii];
          itemRatings[item] = rating;
        }
        
        int nSets = uSet.itemSets.size();
        for (int i = 0; i < nSets; i++) {
          opFile << uSet.user << " ";
          for (auto&& item: uSet.itemSets[i].first) {
            opFile << item << " " << itemRatings[item] << " ";
          }
          opFile << uSet.setsEntropy[i] << std::endl;
        }
      }

      opFile.close();
    }


