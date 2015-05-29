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
      printf("%d ", dUserSet->items[j]);
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


//assuming data is already allocated
void loadData(Data *data, Params *params) {
  
  FILE *fp = NULL;
  char *line = NULL;
  int len = 15000;
  char *token = NULL;
  int read;

  int user, item, setSz, numSets, nUserItems;
  float label;
  int i, j;
  //UserSets *dUserSet; 
  
  int *itemSetInd;

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
      dUserSet->items[i] = item;
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
          dUserSet->itemSetsSize[item] += 1;
        }

      }
    }
     
    //create item to set mapping i.e., itemSets
    for (i = 0; i < nUserItems; i++) {
      //allocate space for storing set indices for item
      item = dUserSet->items[i]; 
      dUserSet->itemSets[item] = (int*) malloc(sizeof(int)*(dUserSet->itemSetsSize[item]));
    }
    
    memset(itemSetInd, 0, sizeof(int)*params->nItems);

    for (i = 0; i < numSets; i++) {
      for (j = 0; j < dUserSet->uSetsSize[i]; j++) {
        item = dUserSet->uSets[i][j];
        dUserSet->itemSets[item] [itemSetInd[item]++] = i;
      }
    }

    memset(line, 0, len);
  }


  fclose(fp);
  if (line) {
    free(line);
  }
  free(itemSetInd);
}




