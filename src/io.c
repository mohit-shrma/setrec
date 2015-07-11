#include "io.h"


void writeData(Data *data) {
  
  int i, j, k;
  UserSets *dUserSet;
    
  for (i = 0; i < data->nUsers; i++) {
    dUserSet = data->userSets[i];
    //print: userId numSets nItems uniqItems 
    printf("%d %d %d ", dUserSet->userId, dUserSet->numSets, 
        dUserSet->nUserItems); 
    for(j = 0; j < dUserSet->nUserItems; j++) {
      printf("%d ", dUserSet->itemWtSets[j]->item);
    }
    printf("\n");
    
    //print: label numItem itemsInSet
    for (j = 0; j < dUserSet->numSets; j++) {
      printf("%d %d ", (int)dUserSet->labels[j], dUserSet->uSetsSize[j]);
      for (k = 0; k < dUserSet->uSetsSize[j]; k++) {
        printf("%d ", dUserSet->uSets[j][k]);
      }
      printf("\n");
    }

  }

}


void RatingSet_load(RatingSet *ratSet, char *ip, int setSize) {
  
  FILE *fp = NULL;  
  char *line = NULL;
  size_t len = 15000;
  char *token = NULL;
  int read, i;
  int user, item;
  float rating;

  RatingSet_init(ratSet, setSize);
   
  line = (char*) malloc(len);
  
  //open set file  
  fp = fopen(ip, "r");
  if (fp == NULL) {
    printf("\nError opening file: %s", ip);
    exit(EXIT_FAILURE);
  }
  
  memset(line, 0, len);
  
  i = 0;
  while ((read = getline(&line, &len, fp)) != -1) {
    
    if (read >= len) {
      printf("\nErr: line > capacity specified");
    }
   
    //tokenize the line
    token = strtok(line, " ");
    
    //get user
    user = atoi(token);
    
    //get item
    item = atoi(strtok(NULL, " "));

    //get rating
    rating = atof(strtok(NULL, " "));
    
    ratSet->rats[i]->user = user;
    ratSet->rats[i]->item = item;
    ratSet->rats[i]->rat = rating;
    i++;
    memset(line, 0, len);
  }

  printf("\nRead %s  size: %d", ip, i);
  
  assert(i == setSize);

  fclose(fp);
  free(line);
}


void RatingSet_write(RatingSet *ratSet) {
  
  int i;
  
  for (i = 0; i < ratSet->size; i++) {
    printf("\n%d %d %f", ratSet->rats[i]->user, ratSet->rats[i]->item, 
        ratSet->rats[i]->rat);
  }

}


//assuming data is already allocated
void loadData(Data *data, Params *params) {
  
  FILE *fp = NULL;
  char *line = NULL;
  size_t len = 15000;
  char *token = NULL;
  int read;

  int user, item, setSz, numSets, nUserItems;
  float label;
  int i, j;
  //UserSets *dUserSet; 
  
  int *itemSetInd;
  int posCount, negCount, zeroCount;
  
  posCount = negCount = zeroCount = 0;

  ItemWtSets *itemWtSets;

  line = (char*) malloc(len);

  Data_init(data, params->nUsers, params->nItems); 

  itemSetInd = (int*) malloc(sizeof(int)*params->nItems);

  //open file
  fp = fopen(params->user_set_file, "r");
  if (fp == NULL) {
    printf("\nError opening file: %s", params->user_set_file);
    exit(EXIT_FAILURE);
  }

  //printf("\nreading file %s ...", params->user_set_file);

  memset(line, 0, len);
  while( (read = getline(&line, &len, fp)) != -1) {
    
    if (read >= len) {
      printf("\nErr: line > capacity specified");
    }
    
    //tokenize the line
    token = strtok(line, " ");
    
    //get user and num sets
    user = atoi(token);

    numSets = atoi(strtok(NULL, " "));
    nUserItems = atoi(strtok(NULL, " "));
    
    //printf("\nUser: %d numSets: %d nUserItems: %d", user, numSets, nUserItems);
    //fflush(stdout);

    //initialize  user sets
    UserSets * const dUserSet = data->userSets[user];
    UserSets_init(dUserSet, user, numSets, params->nItems, nUserItems);
    
    //read uniq artists for the current user
    for (i = 0; i < nUserItems; i++) {
      item = atoi(strtok(NULL, " "));
      dUserSet->itemWtSets[i]->item = item;
      dUserSet->itemWtSets[i]->szItemSets = 0;
      dUserSet->itemWtSets[i]->wt = 0;
    }

    //sort user items 
    UserSets_sortItems(dUserSet); 

    for (i = 0; i < nUserItems; i++) {
      dUserSet->itemWtSets[i]->ind = i;
    }

    for (i = 0; i < numSets; i++) {
      memset(line, 0, len);
      //read labeled sets by user
      if( (read = getline(&line, &len, fp)) != -1) {
        
        //tokenize the line
        //printf("\nline: %s", line);
        token = strtok(line, " ");
        
        label = atof(token);
        dUserSet->labels[i] = label;

        if (label > 0) {
          posCount++;
        } else if (label < 0) {
          negCount++;
        } else {
          zeroCount++;
        }

        setSz = atoi(strtok(NULL, " "));
        dUserSet->uSetsSize[i] = setSz;
       
        //printf("\n label = %f setSize = %d", label, setSz);
        //fflush(stdout);

        dUserSet->uSets[i] = (int*) malloc(sizeof(int)*setSz);
        memset(dUserSet->uSets[i], 0, sizeof(int)*setSz);
        
        for (j = 0; j < setSz; j++) {
          item = atoi(strtok(NULL, " "));
          dUserSet->uSets[i][j] = item;
          //inc set count of item under consideration
          itemWtSets = UserSets_search(dUserSet, item);
          //TODO: make sure init 0
          assert(itemWtSets != NULL);
          itemWtSets->szItemSets += 1;
        }

      }
    }
     
    //create item to set mapping i.e., itemSets
    for (i = 0; i < nUserItems; i++) {
      //allocate space for storing set indices for item
      dUserSet->itemWtSets[i]->itemSets = (int*) malloc(sizeof(int)*(dUserSet->itemWtSets[i]->szItemSets));
    }
    
    memset(itemSetInd, 0, sizeof(int)*params->nItems);

    for (i = 0; i < numSets; i++) {
      for (j = 0; j < dUserSet->uSetsSize[i]; j++) {
        item = dUserSet->uSets[i][j];
        //search for itemWtSet corresponding to item
        itemWtSets = UserSets_search(dUserSet, item);
        itemWtSets->itemSets[itemSetInd[item]] = i;
        itemSetInd[item]++;
      }
    }

    memset(line, 0, len);
  }

  printf("\nposCount: %d, negCount: %d, zeroCount: %d", posCount, negCount, 
      zeroCount);

  fclose(fp);
  
  //load test set
  RatingSet_load(data->testSet, params->test_set_file,  params->test_set_size); 

  //load validation set
  RatingSet_load(data->valSet, params->val_set_file, params->val_set_size);

  if (line) {
    free(line);
  }
  free(itemSetInd);
}


void loadItemSims(Params *params, float **sim) {
  
  FILE *fp = NULL;
  char *line = NULL;
  char *token = NULL;
  size_t len = 200;
  int read;

  int item1, item2;

  fp = fopen(params->ext_setSim_file, "r");
  
  line = (char*) malloc(len);
  memset(line, 0, len);

  if (fp == NULL) {
    printf("\nError opening file: %s", params->ext_setSim_file);
    exit(EXIT_FAILURE);
  }
  
  while ((read = getline(&line, &len, fp)) != -1) {
  
    if (read >= len) {
      printf("\nErr: line > capacity");
    }

    token = strtok(line, " ");
    item1 = atoi(token);
    item2 = atoi(strtok(NULL, " "));
    sim[item1][item2] = atof(strtok(NULL, " "));
  
  }

  fclose(fp);
}


