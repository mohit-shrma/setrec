#include "io.h"


void writeData(Data *data) {
  
  int i, j, k;
  UserSets *dUserSet;
    
  for (i = 0; i < data->nUsers; i++) {
    dUserSet = &(data->userSets[i]);
    //print: userId numSets nItems uniqItems 
    printf("\n%d %d %d  ", dUserSet->userId, dUserSet->numSets, 
        dUserSet->nUserItems); 
    for(j = 0; j < dUserSet->nUserItems; j++) {
      printf("%d ", dUserSet->items[j]);
    }
    printf("\n");
    
    //print: label numItem itemsInSet
    for (j = 0; j < dUserSet->numSets; j++) {
      printf("%f %d ", dUserSet->labels[j], dUserSet->uSetsSize[j]);
      for (k = 0; k < dUserSet->uSetsSize[j]; k++) {
        printf(" %d ", dUserSet->uSets[j][k]);
      }
      printf("\n");
    }

  }

}


//assuming data is already allocated
void loadData(Data *data, Params *params) {
  
  FILE *fp = NULL;
  char *line = NULL;
  char *token = NULL;
  size_t len = 0;
  ssize_t read;

  int user, item, setSz, numSets, nUserItems;
  float label;
  int i, j;
  UserSets *dUserSet; 
  
  int *itemSetInd;

  Data_init(data, params->nUsers, params->nItems); 

  itemSetInd = (int*) malloc(sizeof(int)*params->nItems);

  //open file
  fp = fopen(params->user_set_file, "r");
  if (fp == NULL) {
    printf("\nError opening file: %s", params->user_set_file);
    exit(EXIT_FAILURE);
  }

  while( (read = getline(&line, &len, fp)) != -1) {
    //tokenize the line
    token = strtok(line, " ");
    
    //get user and num sets
    user = atoi(token);
    numSets = atoi(strtok(NULL, " "));
    nUserItems = atoi(strtok(NULL, " "));
    

    //initialize  user sets
    dUserSet = &(data->userSets[user]);
    UserSets_init(dUserSet, user, numSets, params->nItems, nUserItems);
    
    //read uniq artists for the current user
    for (i = 0; i < nUserItems; i++) {
      item = atoi(strtok(NULL, " "));
      dUserSet->items[i] = item;
    }


    for (i = 0; i < numSets; i++) {
      //read labeled sets by user
      if( (read = getline(&line, &len, fp)) != -1) {
        
        //tokenize the line
        token = strtok(line, " ");
        
        label = atoi(token);
        dUserSet->labels[i] = label;

        setSz = atoi(strtok(line, " "));
        dUserSet->uSets[i] = (int*) malloc(sizeof(int)*setSz);
        memset(dUserSet->uSets[i], 0, sizeof(int)*setSz);
        dUserSet->uSetsSize[i] = setSz;
        
        for (j = 0; j < setSz; j++) {
          item = atoi(strtok(line, " "));
          dUserSet->uSets[i][j] = item;
          dUserSet->itemSetsSize[item] += 1;
        }

      }
    }
     
    //create item to set mapping i.e., itemSets
    for (i = 0; i < nUserItems; i++) {
      //allocate space for storing set indices for item
      item = dUserSet->items[i]; 
      dUserSet->itemSets[item] = (int*) malloc(sizeof(int)*dUserSet->itemSetsSize[item]);
    }
    
    memset(itemSetInd, 0, sizeof(int)*params->nItems);

    for (i = 0; i < numSets; i++) {
      for (j = 0; j < dUserSet->uSetsSize[i]; j++) {
        item = dUserSet->uSets[i][j];
        dUserSet->itemSets[item] [itemSetInd[item]++] = i;
      }
    }

  }

  fclose(fp);
  if (line) {
    free(line);
  }
  free(itemSetInd);
}




