# File Information

## ClusterAnalysis.ipynb: 
Describes the approach taken to cluster students based on their persona and some analysis of the results. The results of the clustering algorithm are used in the web application. 

## TransformerModel.ipynb:
Describes the deep learning approach taken to predict the correctness of student repsonses 

## XGBModel.ipynb: 
Describes the machine learning approach taken to predict the future performance of each student. This was the approach eventually implemented in the final web application.

## Network_Analysis_Final.ipynb:
Describes the network analysis used to predict the the learning path for each of the individual cluster. This was used to demonstrate, validate, and recommend the learning paths for each of the individual clusters.
- "labels.csv" contain the results from ClusterAnalysis.ipynb, and is used to tag individual students to their corresponding clusters.

- "na_recommendation" is the folder that contains the scripts used in this notebook to evaluate the network from the individual student clusters as well as the implementation of the learning path algorithm.


```python
from na_recommendation import utils as utils
# obtain this dataset from junyi website
log_problem = pd.read_csv('Log_Problem.csv')
# obtain this dataset from junyi website
info_content = pd.read_csv('Info_Content.csv')
# the level4_ids that determine the number of exercises that can be recommended
level4_ids = info_content['level4_id'].unique()
# choose the type of learning path to recommend
types = ['number_of_individual_students', 'final_average_performance', 'shortest_path']
selected_type = types[0]
# run the learning path recommender, where result is a dictionary, and G is the resulting network
result, G = utils.recommend_learning_paths(log_problem_sub, info_content, level4_ids, method = selected_type)
# store the learning paths in a variable
paths = res['paths']