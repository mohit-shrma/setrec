#include "ModelBaseline.h"


float ModelBaseline::estItemRating(int user, int item) {
  return globalItemRatings[item];
}


void ModelBaseline::train(const Data& data, const Params& params, 
    Model& bestModel) {
  //map containing no. of sets the item appears
  std::map<int, int> itemSetCount;
  std::cout << "size of globalItemRatings: " << globalItemRatings.size() << std::endl; 
  //go over train sets and add to item the rating given to the set
  for (auto&& uSet: data.trainSets) {
    for (auto&& itemSet: uSet.itemSets) {
      auto items = itemSet.first;
      auto rating = itemSet.second;
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
    } 
  }
  
  //update global item ratings w average
  for (auto&& kv: globalItemRatings) {
    int item = kv.first;
    globalItemRatings[item] = globalItemRatings[item]/itemSetCount[item];
  }
  
  std::cout << " train ratings RMSE: " << rmse(data.trainSets, data.ratMat) 
    << " test ratings RMSE: " << rmse(data.testSets, data.ratMat)
    << " recall@10: " << recallTopN(data.ratMat, data.trainSets, 10)
    << " spearman@10: " << spearmanRankN(data.ratMat, data.trainSets, 10)
    << " inversion count@10: " << inversionCount(data.ratMat, data.trainSets, 10)
    << std::endl;
  bestModel = *this;
}


