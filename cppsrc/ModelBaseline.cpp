#include "ModelBaseline.h"


float ModelBaseline::estItemRating(int user, int item) {
  //return globalItemRatings[item];
  return uSetBias(user);
}


float ModelBaseline::estSetRating(int user, std::vector<int>& items) {
  return uSetBias(user);
}


void ModelBaseline::train(const Data& data, const Params& params, 
    Model& bestModel) {
  
  uSetBias.fill(0);

  trainUsers = data.trainUsers;
 
  for (int item = 0; item < nItems; item++) {
    if (data.partTrainMat->colptr[item+1] - data.partTrainMat->colptr[item] > 0) {
      trainItems.insert(item);
    }
  }
  //trainItems = data.trainItems;
  
  std::cout << "no trainItems: " << trainItems.size() << std::endl;

  //map containing no. of sets the item appears
  std::map<int, int> itemSetCount;
  std::cout << "size of globalItemRatings: " << globalItemRatings.size() << std::endl; 
 
  float meanItemRating = meanRating(data.partTrainMat);
  float meanSet = 0, setCount = 0;
  float uMean = 0;
  //go over train sets and add to item the rating given to the set
  for (auto&& uSet: data.trainSets) {
    int user = uSet.user;
    uMean = 0;
    for (auto&& itemSet: uSet.itemSets) {
      auto items = itemSet.first;
      auto rating = itemSet.second;
      meanSet += rating;
      uMean += rating;
      setCount++;
      /*
      for (auto&& item: items) {
        //update rating map for the item
        if (globalItemRatings.find(item) == globalItemRatings.end()) {
          globalItemRatings[item] = 0;
        }
        globalItemRatings[item] += rating;

        //update set count map for the item
        if (itemSetCount.find(item) == itemSetCount.end()) {
          itemSetCount[item] = 0;
        }
        itemSetCount[item]++;
      }
      */
    }
    uMean = uMean/uSet.itemSets.size();
    uSetBias(user) = uMean;
  }
  meanSet = meanSet/setCount;
  
  std::cout << "meanSet: " << meanSet << " meanItem: " << meanItemRating << std::endl;
  auto meanItemRatings = itemAvgRating(data.partTrainMat);  
  auto meanSubRatings = meanSubtractedItemRating(data.partTrainMat, meanSet);
  //update global item ratings w average
  for (int item = 0; item < globalItemRatings.size(); item++) {
    globalItemRatings[item] = meanItemRatings[item];//meanSubRatings[item];
    //globalItemRatings[item] = meanSubRatings[item];
    //globalItemRatings[item] = globalItemRatings[item]/itemSetCount[item];
  }
  
  std::cout << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
    << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
    << std::endl;
  bestModel = *this;
}


