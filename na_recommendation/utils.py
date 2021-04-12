import pandas as pd
import numpy as np
import networkx as nx

def preprocess_data(log_problem):
  """
  NOTE: can you do this inside the backend so that we do not need to run this in runtime?

  1. create 'ucid2' which is the numeical_id of ucid
  2. sort our log_problem data according to time
  3. change 'is_correct' from boolean to int
  4. only return the fields that are required: uuid, ucid, is_correct, 
  """
  # convert hashes into numerical index
  ucid_map = {ucid:index+1 for index, ucid in enumerate(info_content['ucid'].value_counts().sort_index().index)}
  log_problem['ucid2'] = log_problem['ucid'].map(ucid_map)
  log_problem.sort_values(by=['timestamp_TW'], inplace=True)
  log_problem['is_correct'] = log_problem['is_correct'].apply(lambda x: 1 if x==True else 0)
  return log_problem

def filter_log_problem_by_selected_exercise_id(log_problem, info_content, level4_ids):
  """
  Filter the log_problem data at the beginning so that the run_time will be faster later on

  Parameters
  ----------
  log_problem : pd.DataFrame
    Must be the data that has the log info

  info_content: pd.DataFrame
    Must be the data that has the content information

  lelve4_ids : list
    the list of level4 ids that are being selected, that will be used to create the exercise ids

  Returns
  -------
  log_problem_filtered : pd.DataFrame
    The log_problem that only contains the exercises inside input exercise_ids
  """
  exercise_ids = info_content['ucid'][info_content['level4_id'].isin(level4_ids)]
  return log_problem[log_problem['ucid'].isin(exercise_ids)]


def lower_bound_student_exercise_frequencies(log_problem, lower_bound=4):
  """
  Remove the students who took the exercises less than "lower_bound" times.
  For example, if a student only took one problem_id in the exercise, this will not correctly reflect the accuracy of the exercise itself.

  Parameters
  ----------
  log_problem : pd.DataFrame
    The log_problem that is filtered according to the exercise_ids that we want

  lower_bound : int
    The minimum number of problem_ids that a student must take in order to contribute to the mean accuracy of the exercise.
  """
  filtered_df = log_problem.groupby(['uuid','ucid'], sort=False)['upid'].count()
  filtered_df = filtered_df[filtered_df>lower_bound].reset_index().drop(columns=['upid'], axis=1)
  log_problem_filtered = log_problem.merge(filtered_df, left_on=['uuid','ucid'], right_on=['uuid','ucid'], how='right')
  return log_problem_filtered

def create_graphing_dataframe(flp):
  """
  Using the flp, we want to create a list of src nodes: "from", a list of tgt nodes: "to", and a list of the subsequent individual scores to tgt: "individual_sum_score_to"

  Parameters
  ----------
  flp: pd.DataFrame
    The dataframe must contain uuid, ucid, ucid2, is_correct
  
  Return
  ------
  user_to_content_student_performance : pd.DataFrame
    uuid, ucid2, list of src nodes: "from", a list of tgt nodes: "to", and a list of the subsequent individual scores to tgt: "individual_sum_score_to"
  """
  # create SUM student performance for each student on a particular exercise
  student_sum_performance = flp.groupby(['uuid','ucid'], sort=False)['is_correct'].sum().reset_index().rename(columns={'is_correct':'sum_is_correct'})
  # Remove duplications of students and exercise since we aggregated the result in the df above
  flp_student_sum_performance = flp.drop_duplicates(subset=['uuid','ucid']).merge(student_sum_performance, left_on=['uuid','ucid'], right_on=['uuid','ucid'])

  # create the dataframe to create our graph
  user_to_content_student_performance = flp_student_sum_performance.groupby(['uuid'], sort=False)['ucid2'].apply(lambda x: x.tolist()).reset_index()
  user_to_content_student_performance['individual_sum_score'] = flp_student_sum_performance.groupby(['uuid'], sort=False)['sum_is_correct'].apply(lambda x: x.tolist()).values

  # # need to remove students who only took one exercise, since they do not contribute to edge weight
  user_to_content_student_performance = user_to_content_student_performance[user_to_content_student_performance['ucid2'].apply(lambda x:len(x)>1)]
  user_to_content_student_performance['from'] = user_to_content_student_performance['ucid2'].apply(lambda x: x[:-1])
  user_to_content_student_performance['to'] = user_to_content_student_performance['ucid2'].apply(lambda x: x[1:])
  user_to_content_student_performance['individual_sum_score_to'] = user_to_content_student_performance['individual_sum_score'].apply(lambda x: x[1:])
  return user_to_content_student_performance[['uuid','from','to','individual_sum_score_to']]


def create_networkx_graph(user_to_content_student_performance):
  """
  Create the graph according to the given inputs that contains uuid, from, to, individual_sum_score_to

  Parameters
  ----------
  user_to_content_student_performance : pd.DataFrame
    The dataframe that we created using create_graphing_dataframe(flp)

  Returns
  G: NetworkXGraph
    The graph has edge attributes:
      1. number_of_individual_students from src to target node
      2. sum_of_correct_problems from src to target node
      3. average_score_from_src_to_tgt (part A in diagram)
      4. average_performance_tgt (Part B in diagram)
      5. final_average_performance (Part C in diagram)
  """
  G = nx.DiGraph()
  for index, row in user_to_content_student_performance.iterrows():
      user = row['uuid']
      from_list = row['from']
      to_list = row['to']
      individual_sum_score_to = row['individual_sum_score_to']
      # update the edges weights
      for src, tgt, score in zip(from_list, to_list, individual_sum_score_to):
        if G.has_edge(src, tgt):
            G[src][tgt]['number_of_individual_students'] += 1
            G[src][tgt]['sum_of_correct_problems'] += score
        else:
            G.add_edge(src, tgt, sum_of_correct_problems=score, number_of_individual_students=1)
  return get_true_edge(G)


def get_true_edge(G):
  """
  Using the auxillary edge weights, create the final weight as shown in the diagram above

  Parameters
  ----------
  G: NetworkXGraph

  Returns
  -------
  G: NetworkXGraph
    This graph now has additional edge attributes :
      1. average_score_from_src_to_tgt (part A in diagram)
      2. average_performance_tgt (Part B in diagram)
      3. final_average_performance (Part C in diagram)
      4. shortest path: from one src to tgt
  """
  # Add the shortest path edge weights
  path_src_to_tgt_dic = {(u,v):1 for u,v,dic in G.edges(data=True)}
  nx.set_edge_attributes(G, path_src_to_tgt_dic, 'shortest_path')

  # calculate the average performance. handles part A
  average_score_from_src_to_tgt_dic = {(u,v): (dic['sum_of_correct_problems']/dic['number_of_individual_students']) for u,v,dic in G.edges(data=True)}
  nx.set_edge_attributes(G, average_score_from_src_to_tgt_dic, 'average_score_from_src_to_tgt')
  
  # calculate the average performance of the target. handles part B
  def calc_average_performance_on_tgt(node):
    lst = list(map(lambda x: x[2],G.in_edges(node, data='average_score_from_src_to_tgt')))
    return np.mean(lst) if len(lst) > 0 else 0
  average_performance_on_tgt = {node:calc_average_performance_on_tgt(node) for node in G.nodes()}
  nx.set_node_attributes(G, average_performance_on_tgt, 'average_performance_tgt')

  # calculate the actual weight according to part C
  final_average_performance_dic = {(u,v): (w - G.nodes[v]['average_performance_tgt']) for u,v,w in G.edges(data='average_score_from_src_to_tgt')}
  nx.set_edge_attributes(G, final_average_performance_dic, 'final_average_performance')
  
  return G


def get_hubs_and_authorities(G):
  """
  Calculate the Hubs and Authority scores to identify potential starting and ending exercises

  Parameters
  ----------
  G: NetworkXGraph

  Returns
  -------
  authorities_hubs: pd.DataFrame
    exercises are ranked according to the highest Hub score and the highest Authority score. 
  """
  names = ["Authorities", "Hubs" ]
  # handle convergence of "hits"
  iterate = True
  max_iter = 100
  iter_num = 1
  hits = False
  while hits==False and iter_num <=50:
    try:
      iter_num += 1
      max_iter+= 100
      if iter_num > 0 and iter_num%100 == 0:
        print(f'Current iter_num for get_hubs_and_authorities: {iter_num}, max_iter: {max_iter}')
      hits = nx.hits(G, max_iter=max_iter) # https://stackoverflow.com/questions/63026282/error-power-iteration-failed-to-converge-within-100-iterations-when-i-tried-t
    except nx.PowerIterationFailedConvergence:
      continue
  if hits == False:
    print(f'Hits not found for graph of num_node: {len(G)}, num_edges: {len(G.edges)}')
    # if hits did not converge, return back a dummy column (so it will be recommending a random exercise.. this)
    df = pd.DataFrame({'Authorities':[0]*len(G),'Hubs':[0]*len(G)}, index = [node for node in G.nodes()])
  else:
    # if the hits algorithm converged, get the proper hits values
    all_measures = [hits[1], hits[0]]
    df = pd.concat([pd.Series(measure) for measure in all_measures], axis=1)
    df.columns = names
  authorities_hubs = df[['Authorities','Hubs']].sort_values(by='Authorities')
  authorities_hubs['Authorities_Rank'] = authorities_hubs['Authorities'].rank(ascending=False)
  authorities_hubs['Hubs_Rank'] = authorities_hubs['Hubs'].rank(ascending=False)
  return authorities_hubs 


def get_top_k_hubs_and_authorities(df, k=5):
  """
  Return the list of potential starting exercises and list of ended exercises

  Parameters
  ----------
  df : pd.DataFrame
    The df that contains the corresponding Hub and Authority scores with their ranks attached to each exercise.
  
  k : int
    How many starting and ending exercises do we want to consider

  Returns
  -------
  top_k_hubs: list
    list of k potential starting exercises

  top_k_authorities: list
    list of k potential ending exercises
  """
  top_k_hubs = df[df['Hubs_Rank']<=k]
  top_k_authorities = df[df['Authorities_Rank']<=k]
  return top_k_hubs, top_k_authorities


def calculate_path_weight(G, path, method='final_average_performance'):
  """
  Parameters
  ----------
  G: NetworkXGraph

  path : list
    An actual path eg [1,2,3]
  
  method : str
    the edge weight used to calculate the path weights: can be any of the edge attributes mentioned in create_networkx_graph()

  Returns
  -------
  weight : float
    The sum of the the total path cost from start to ending exercise
  """
  weight = 0
  weight_list = []
  for u,v in zip(path[:-1],path[1:]):
    weight += G[u][v][method]
    weight_list.append(G[u][v][method])
  return weight_list
  

def k_shortest_paths(G,source, target, num_paths=1, minimum_path_length=5, method= 'number_of_individual_students', relax=True):
  paths_to_return = []
  # store the original method to calculate the path weights
  orig_method = method
  # store the method name prefixed with "neg_" which is the name of the preprocessed edge weights for clarity.
  # For the "shortest_path", the name is not prefixed.
  method = 'neg_' + orig_method if orig_method != "shortest_path" else orig_method
  # generator that recommends paths
  path_generator = nx.shortest_simple_paths(G, source, target, weight=method)
  # keep track of the number of rejected paths because the length is too short
  num_rejected = 0
  # keep finding paths until we get the desired number of paths
  while len(paths_to_return) < num_paths:
    try:
      candidate_path = next(path_generator)
      # if the method to use the shortest path is "shortest_path", set a different rejection threshold to find the path_length
      if method == "shortest_path":
        # limit the length of the path to 2* the minimum path length
        if len(candidate_path) >= minimum_path_length and len(candidate_path) <= 2*minimum_path_length:
          path_cost = calculate_path_weight(G, candidate_path, orig_method)
          paths_to_return.append((candidate_path, path_cost))
        else:
          # if we want to relax the length of the path recommended
          if relax:
            num_rejected+=1
            # the number of rejected value is too much decrease the path length to try and return something (this is the part that is different from the ones below)
            if num_rejected >= 500:
              # ensure no negative path length
              minimum_path_length = max(0, minimum_path_length-1)
              # if reached the limit, break out of the loop and return the incomplete list of paths
              if minimum_path_length == 0:
                break
      else:
        # limit the length of the path to 2* the minimum path length
        if len(candidate_path) >= minimum_path_length and len(candidate_path) <= 2*minimum_path_length:
          path_cost = calculate_path_weight(G, candidate_path, orig_method)
          paths_to_return.append((candidate_path, path_cost))
        else:
          # if we want to relax the length of the path recommended
          if relax:
            num_rejected+=1
            # the number of rejected value is too much, decrease the path length to try and return something
            if num_rejected >= 10:
              # ensure no negative path length
              minimum_path_length = max(0, minimum_path_length-1)
              # if reached the limit, break out of the loop and return the incomplete list of paths
              if minimum_path_length == 0:
                break
    except StopIteration:
      break
  return paths_to_return

def convert_edge_weights_for_dijstra(G, method='number_of_individual_students'):
  # find the maximum weight for this 'method' to make our negated_weights all > 0
  offset = np.max(list(map(lambda x: x[2], G.edges(data=method)))) + 2
  negated_weights = {(u,v):-w + offset for u,v,w in G.edges(data=method)}
  # overwrite the original name with this new negated weights
  nx.set_edge_attributes(G, negated_weights, f'neg_{method}')
  return G

def preprocess_edge_weights(G, method):
  if method == 'shortest_path':
    # if shortest path, we do not need to change the edge weights, default it will be one anyway.
    return G
  else:
    # if it is popularity or student performance, we need to negate these edge weights and make them positive edges
    G = convert_edge_weights_for_dijstra(G, method)
    return G

def get_at_least_k_paths(G, top_k_hubs, top_k_authorities, num_paths = 3,   minimum_path_length = 5, method = 'final_average_performance'):
  """
  Parameters
  ----------
  G: NetworkXGraph
  
  top_k_hubs : list
    list of k potential starting exercises

  top_k_authorities : list
    list of k potential ending exercises

  num paths : int
    The maximum number of paths to store and recommend

  minimum_path_length : int
    The minimum path length for each of the returned paths
  
  method : str
    The edge weight to be used for calculating the path cost as well as pruning the network

  Returns
  -------
  paths_to_return: list of list
    The list contains up to k paths
  """
  paths_to_return = []
  # convert edges that should be negated into their negative values, and then scaled accordingly.
  G = preprocess_edge_weights(G, method)
  for hub in top_k_hubs.index:
    for aut in top_k_authorities.index:
      if hub != aut:
        try:
          # if we get the number of paths already, terminate
          if len(paths_to_return) >= num_paths:
              return paths_to_return

          # Store paths that are shortest according only to the 'method'
          if len(paths_to_return)<num_paths:
            # just get 1 path to recommend a variety of starting and ending exercises
            paths = k_shortest_paths(G, hub, aut, num_paths=2, minimum_path_length=minimum_path_length, method=method)
            # append this path into our resulting list
            paths_to_return.extend(paths)

        except nx.NetworkXNoPath:
          continue
  return paths_to_return

def prune_by_quantile(G, percentiles, criterion = 'final_average_performance', num_hubs_auth=10, minimum_number_of_paths=5, minimum_path_length=5):
  """
  Helper for tune_get_at_least_k_paths().

  Prune the network edges directly using the "criterion", desired number of potential starting and ending exercise, 
  until we get the desired minimum number of path with the desired minimum number of length. 

  This code will prune the edge weights according to the percentiles for the edge weights.
  It will loop through this percentiles and see if any of the percentiles allow us to attain the desired number of path and path length.

  If this percentile still does not give us the desired number of paths and path length, it will be determinated with error message.

  {'Success': False,
            'paths': best_paths,
            'best_quantile': best_quantile,
            'ErrorMessage':'Something went wrong :('}

  Parameters
  ----------
  G: NetworkXGraph
  
  percentiles: list
    The list of percentiles of the edge weights to try for pruning

  criterion : str
    The edge weight to be used for calculating the path cost as well as pruning the network

  num_hubs_auth : int
    The number of potential hubs and authorities to choose

  minimum_number_of_paths : int
    The minimum number of paths to return

  minimum_path_length : int
    The minimum path length for each of the returned paths

  Returns
  -------
  dictionary
   - {'Success': bool,
      'paths': list,
      'best_quantile': float,
      'ErrorMessage': str}
  """
  attribute_data = pd.Series(map(lambda x: x[2], G.edges(data=criterion)))
  best_length = -1
  best_quantile = -1
  best_paths = []
  for quantile in percentiles:
    # create a copy of the original graph for copy, so the overwrite of the edge weights when recommending paths will not affect the original graph
    F = slice_network_edges_by(G, by=criterion, copy=True, lower_limit = attribute_data.quantile(quantile))
    try:
      df = get_hubs_and_authorities(F)
    except ZeroDivisionError:
      # if all the edges are removed, then continue to next quantile
      print('DivisonError on prune_by_quantile')
      continue

    top_k_hubs, top_k_authorities = get_top_k_hubs_and_authorities(df, k=num_hubs_auth)

   # paths are recommended through the copy of the graph 
    paths = get_at_least_k_paths(F, top_k_hubs, top_k_authorities, num_paths=minimum_number_of_paths, minimum_path_length = minimum_path_length, method=criterion)

    if len(paths)>=minimum_number_of_paths:
      satisfied=True
      return {'Success':True,
                'paths':paths,
                'ErrorMessage':''}
    else:
      if len(paths) > best_length:
        best_length = max(len(paths), best_length)
        best_quantile = quantile
        best_paths = paths
  
  return {'Success': False,
          'paths': best_paths,
          'best_quantile': best_quantile,
          'ErrorMessage':'Something went wrong :('}

def slice_network_edges_by(G, by, copy=False, lower_limit=None, upper_limit=None):
  """
  Remove the edges that have {by} less than the specified lower_limit or more than the specified upper_limit

  Parameters
  ----------
  G: NetworkXGraph
  
  by: str
    The edge attribute to use to prune the edges

  copy : bool
    True if want inplace else False

  lower_limit: float
    The minimum edge weight you want to be present in the network

  upper_limit: float
    The maximum edge weight you want to be present in the network

  Returns
  -------
  F: NetworkXGraph
  """
  # if it is the "shortest_path", prune the network by the number of individual students
  # if prune by shortest path, every path will be pruned in our function prune_by_quantile() since the percentiles are the same.
  if by == 'shortest_path':
    by = "number_of_individual_students"
  F = G.copy() if copy else G
  if not lower_limit is None: 
    F.remove_edges_from((u,v) for u, v, att in list(F.edges(data=by)) if att <= lower_limit)
  elif not upper_limit is None:
    F.remove_edges_from((u,v) for u, v, att in list(F.edges(data=by)) if att >= upper_limit)
  return F

def tune_get_at_least_k_paths(G, method='final_average_performance', num_hubs_auth = 5, minimum_number_of_paths=5, minimum_path_length=5, ):
  """
  The code tries to prune according to 10 quantiles from 0.01 to 0.99. 

  This code will prune the edge weights according to the percentiles for the edge weights.
  It will loop through this percentiles and see if any of the percentiles allow us to attain the desired number of path and path length.

  If it doesnt, it will create a new list of percentiles to try:
    This new list of percentiles is created using the percentile that gave the most number of paths, and adds +0.2 and -0.2 to determine the upper and lower bound of the list.

  If this percentile still does not give us the desired number of paths and path length, it will be determinated with error message.

  Parameters
  ----------
  G: NetworkXGraph
  
  criterion : str
    The edge weight to be used for calculating the path cost as well as pruning the network

  num_hubs_auth : int
    The number of potential hubs and authorities to choose

  minimum_number_of_paths : int
    The minimum number of paths to return

  minimum_path_length : int
    The minimum path length for each of the returned paths

  Returns
  -------
  dictionary
   - {'Success': bool,
      'paths': list,
      'best_quantile': float,
      'ErrorMessage': str}

  if no paths found:
    {'Success':False,
                'paths':res['paths'],
                'ErrorMessage':\
                f'Cant find any path, returning the individual nodes.'}

  if at least some path is found:
    {'Success': True,
              'paths':res['paths'],
              'ErrorMessage':\
              f'We can only return paths of minimum_length: {min_path_length}, and we can only find {len(res["paths"])} of them.'}

  if all the desired paths found:
    {'Success':True,
                'paths':paths,
                'ErrorMessage':''}

  """
  # if there are only 2 exercises, you can only go from one exercise to other exercise
  if len(G) == 2:
    df = get_hubs_and_authorities(G)
    top_k_hubs, top_k_authorities = get_top_k_hubs_and_authorities(df, k=minimum_number_of_paths)
    paths = get_at_least_k_paths(G, top_k_hubs, top_k_authorities, num_paths=1, minimum_path_length = 0, method=method)    
    res = {'Success': True,
              'paths':paths,
              'ErrorMessage':\
              f'Length of graph is only 2, special case.'}
    return res

  # prune G by the quantile and get the recommended path, lower bound is 25th percentile and the upper bound is 95th percentile
  res = prune_by_quantile(G, 
                          percentiles=np.linspace(0.10, 0.99,10)[::-1], 
                          criterion = method,
                          num_hubs_auth=num_hubs_auth,
                          minimum_number_of_paths=minimum_number_of_paths,
                          minimum_path_length=minimum_path_length)
  if res['Success'] == True:
    return res

  # if cannot find the desired path, try again on the quantile value that gave us the best result so far (in terms of number of paths)
  else:
    # iterate for another 2 rounds using iter
    iter = 0
    while res['Success'] == False and iter <= 2:
      iter += 1
      delta = 0.2
      # delta- range from the best _quantile, lower bound is 25th percentile and the upper bound is 95th percentile
      range_ = np.hstack((res['best_quantile'],np.linspace(max(0.1, res['best_quantile']-delta),min(0.99,res['best_quantile']+delta),10)))[::-1]
      # prune G by the quantile and get the recommended path
      res = prune_by_quantile(G, 
                              percentiles=range_, 
                              criterion = method, 
                              num_hubs_auth=5, 
                              minimum_number_of_paths=minimum_number_of_paths, 
                              minimum_path_length=minimum_path_length)
    # return the result if the path is found                          
    if res['Success']==True:
      return res

    # if not found, return dictionary accordingly
    else:
      min_path_length = min([len(path) for path, cost in res['paths']]) if len(res['paths']) > 0 else 0
      
      # if we cant find any path, then we return success = False
      if min_path_length == 0:
        res['paths'] = [(node,0) for node in G.nodes()] # return individual exercises if no paths are found
        return {'Success':False,
                'paths':res['paths'],
                'ErrorMessage':\
                f'Cant find any path, returning the individual nodes.'}
      else:
        # if we are not returning  empty paths, consider it a success but return an ErrorMessage nonetheless
        # Eg if we ask for 10 minimum paths of length 5, and the code returns only 5 minimum paths of length 5, we call it a success.
        return {'Success': True,
              'paths':res['paths'],
              'ErrorMessage':\
              f'We can only return paths of minimum_length: {min_path_length}, and we can only find {len(res["paths"])} of them.'}

def run_pipeline(log_problem, info_content, level4_ids, num_hubs_auth):
  """
  Run the entire process
  1. Filter according to the selected exercise
  2. Prune the network by removing students who took only a few problem_id in a exercise
  3. Create the dataframe for graphing
  4. Create the graph
  5. get the potential starting and ending exercises

  Parameters
  ----------
  log_problem: pd.DataFrame
    if recommending for individual clusters(student personas), just filter the log_problem for that specific group/persona

  info_content: pd.DataFrame
    if recommending for individual clusters(student personas), , this has to REMAIN AS THE ENTIRE info_content, since we need this to get the exercises that belongs to the user specified level4_id

  level4_ids: list
    The list of id. This is the original level4_ID , eg '364ml6jwsO0pO5l86JBpC+KFYvYr7mn7S9gVuhoBnUE='

  num_hubs_auth: int
    (REMOVED FOR THE TIME BEING)
    The parameter to control how many potential starting and ending exercises you want

  lower_bound: (REMOVED)
    The parameter to control the number of edges in the network by removing students who took less than lower_bound number of problem in an exercise.

  Returns
  -------
  G: NetworkXGraph

  top_k_hubs : list
    list of k potential starting exercises

  top_k_authorities : list
    list of k potential ending exercises
  """
  flp = filter_log_problem_by_selected_exercise_id(log_problem, info_content, level4_ids)
  student_upid_frequencies = flp.groupby(['uuid','ucid'])['upid'].count()
  # if students only take 0.05 * (number of problems attemped by all students) problems for one exercise, filter those data
  flp = lower_bound_student_exercise_frequencies(flp, lower_bound=student_upid_frequencies.quantile(0.05)) if len(flp['ucid'].unique())!=1 else flp
  user_to_content_student_performance = create_graphing_dataframe(flp)
  G = create_networkx_graph(user_to_content_student_performance)
  return G
  # df = get_hubs_and_authorities(G)
  # top_k_hubs, top_k_authorities = get_top_k_hubs_and_authorities(df,k=num_hubs_auth)
  # return G, top_k_hubs, top_k_authorities

def recommend_learning_paths(log_problem, info_content, level4_ids, num_paths=5, min_path_length=5, method = 'final_average_performance'):
  """
  Recommend the learning paths.

  Parameters 
  ----------
  log_problem: pd.DataFrame
    if recommending for individual clusters(student personas), just filter the log_problem for that specific group/persona

  info_content: pd.DataFrame
    if recommending for individual clusters(student personas), , this has to REMAIN AS THE ENTIRE info_content, since we need this to get the exercises that belongs to the user specified level4_id

  level4_ids: list
    The list of id. This is the original level4_ID , eg '364ml6jwsO0pO5l86JBpC+KFYvYr7mn7S9gVuhoBnUE='

  num_paths: int
    The maximum number of paths that you want to recommend
  
  num_hubs_auth: int
    The parameter to control how many potential starting and ending exercises you want

  student_frequency_lower_bound: int (REMOVED)
    The parameter to control the number of edges in the network by removing students who took less than lower_bound number of problem in an exercise.

  Verbose: bool
    True if want to debug the number of learning paths being recommended else False
  
  Returns
  -------
  paths: list of tuple
    returns the list of recommend path so that we can sort according to the path_cost Eg [(path1, path1_cost), (path2, path2_cost)]

  """
  G = run_pipeline(log_problem, info_content, level4_ids, num_hubs_auth=num_paths*10)

  # if less than min_path_length or 1 exercise(the graph will be empty), return back single exercises
  if len(G)<=min_path_length or len(G)<=1:
      return list(info_content[info_content['level4_id'].isin(level4_ids)]['ucid'].unique())
 
  res = tune_get_at_least_k_paths(G, method = method, minimum_number_of_paths=num_paths, minimum_path_length=min_path_length)
  return res, G
