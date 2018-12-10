# ml_RandomForest
Simple RandomForest implementation written in C++.

Inside it creates binary trees, uses bagging, and calculates split points based on Gini impurity.

Currently does not support categorial features, so you can try to use one-hot encoding instead.

## Expected dataset format is:
```bash
<goal 1> <feature 1 1> <feature 1 2> .. <feature 1 M>
.
.
<goal N> <feature N 1> <feature N 2> .. <feature N M>
````

`<goal i>` , `<feature i j>` - are expected to be int or float, *string values are currently unsupported*.
    
**_!!!Even in prediction mode the first column is expected to be goals!!!_**
    
### Arguments example:
```bash
RForestClassificator.exe -featuresPath "..\..\DataSets\np.txt" -mode cv 1 3 -treeCount 6 -threadCount 6 -oob -shuffle
````
## Command line arguments:
#### File to save\load from model.
```bash
-modelPath <file path>
```
#### Dataset file.
```bash
-featuresPath  <file path>
```
#### File to output predictions.
```bash
-predictionPath  <file path>
```
#### Mode of operation.
```bash
-mode <string>
```

##### possible values:   
1) `"learn"` : Train and save model.

    requires `-modelPath` and `-featuresPath`

2) `"predict"` : Load model and predict on given features.

    requires `-modelPath` , `-featuresPath` and `-predictionPath`

3) `"cv" <rounds> <folds>` : Performs <rounds>*<folds> cross-validation and just prints mean accuracy.
  
      requires `-featuresPath`.
      
    `<rounds>`(=int, default=1)
  
    `<folds>`(=int, default=2)
  

#### Number of trees to create.
```bash
-treeCount <int, default=1>
```

#### Max tree depth.
```bash
-depth <int, default=3>
```

#### Number of threads used when creating trees.
```bash
-threadCount <int, default=1>
```

#### Set to print out of bag error for each tree
```bash
-oob
```

#### Set to shuffle subset of features on each node of the tree, when finding the best split.
```bash
-shuffle
```
#### Define number of features used to find split points at each node.
```bash
-maxNodeFeatures <string> 
```
##### possible values (default=`"sqrt"`):   
1) `"all"` : Use all available features.
2) `"sqrt"` : Use sqare of all available features.
3) `"log"` : Use log of all available features.
4) `"float" <float>` : Use part of features equal to (availble feature count)*`<float>`.  (0<`<float>`<= 1, default=1)
   
#### Define number of features available to tree on construction.
```bash
-featureSubset <string>
```
##### possible values (default=`"all"`):   
1) `"all"` : Use all features from dataset.
2) `"sqrt"` : Use sqare all features from dataset.
3) `"log"` : Use log of all features from dataset.
4) `"float" <float>` : Use part of features equal to (dataset feature count)*`<float>`.  (0<`<float>`<= 1, default=1)

#### Define random type.
```bash
-random <string>
```
##### possible values (default=`"seed" 1`): 
1) `"time"` : Use time since epoch as random seed - udetermined random.
2) `"seed" <int>` : Use `<int>` as random seed - determined random.

# Visualisation:
Python script visualiser.py

Requires graphviz folder in the same catalog (see os.environ["PATH"] extension inside visualiser.py) https://graphviz.gitlab.io/_pages/Download/Download_windows.html

Prints to pdf and opens created files of all trees in model.

Launch example:
```bash
 python .\visualizer.py --model_path=".\model.txt" --limit=10
````
*parameter limit specifies depth of visualisation, if use see blank graph, try decreasing it.
