import pandas as pd
import networkx as nx
from networkx.algorithms import community #This part of networkx, for community detection, needs to be imported separately.
from operator import itemgetter

import functools
import time

def timer(func):
    """
    Some helper function to time the run time.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(
            repr(func.__name__), round(run_time, 3)))
        return value
    return wrapper

@timer
def create_undirected_graph(data):
  G = nx.Graph()
  for index, row in data.iterrows():
    user = row['uuid']
    from_list = row['from']
    to_list = row['to']
    # always check that the number is the same
    # assert len(from_list) == len(to_list)
    for src, tgt in zip(from_list, to_list):
      if G.has_edge(src,tgt):
        G[src][tgt]['weight'] += 1
      else:
        G.add_edge(src, tgt, weight=1)
  return G

@timer
def create_directed_graph(data):
  G = nx.DiGraph()
  for index, row in data.iterrows():
    user = row['uuid']
    from_list = row['from']
    to_list = row['to']
    # always check that the number is the same
    # assert len(from_list) == len(to_list)
    for src, tgt in zip(from_list, to_list):
      if G.has_edge(src,tgt):
        G[src][tgt]['weight'] += 1
      else:
        G.add_edge(src, tgt, weight=1)
  return G

class CompareCentralityMeasures:
  """
  Class to compare the centrality measures across a Original graph and the Graph of the Clusters
  We assume that the input graph to this class is the current Graph, i.e if we want to compare to other Graphs, need to input another graph to compare against this Graph. See self.run().
  """
  def __init__(self):
    self.G = None
    self.name = None
    self.df = None

  def load_graph(self, G, name):
    self.G =G
    self.name = name
    return self

  def compute_graph_centrality_measures(self, G):
    """
    Compute the centrality measures for a graph: ['degree','closeness','betweenness','eigenvector']
    Returned in the form of list of node-val dictionary for each measure
    """
    degree_dict = dict(G.degree(G.nodes()))
    betweenness_dict = nx.betweenness_centrality(G) # Run betweenness centrality
    try:
      eigenvector_dict = nx.eigenvector_centrality(G)
    except nx.PowerIterationFailedConvergence:
      eigenvector_dict = nx.eigenvector_centrality(G, max_iter=500)

    closeness_dict = nx.closeness_centrality(G)
    return [degree_dict, closeness_dict, betweenness_dict, eigenvector_dict]

  def compute_measure_diff(self, own_dict, other_dict):
    """
    Compute the difference betweeen a dictionary of measure with another
    Returns a real valued percentage number. The right dictionary values are used as the base.
    """
    total_difference = 0
    other_dic_total = 0
    for node, val in own_dict.items():
      total_difference += abs(val - other_dict[node])
      other_dic_total += other_dict[node]
    # return the difference as % of the og graph
    return round(total_difference/other_dic_total,2)

  def compare_measures(self, own_measures_dicts, other_measures_dicts):
    """
    Compares the the list of measure between one list of measure and another list of measure
    Returns a list of the the differences in each measures
    """
    measure_diffs = []
    for own_dict, other_dict in zip(own_measures_dicts, other_measures_dicts):
      diff = self.compute_measure_diff(own_dict, other_dict)
      measure_diffs.append(diff)
    return measure_diffs

  def store_centrality_measures(self, df, measure_diffs, col_name):
    """
    Stores the list of differences of the measures under the col name
    Returns a pd.DataFrame
    """
    if self.df is None:
      self.df = pd.DataFrame(['degree','closeness','betweenness','eigenvector'], columns=['index']).set_index('index')
    self.df[col_name] = measure_diffs
    return self.df

  def save_csv(self, name):
    self.df.to_csv(name, index=False)
  
  @timer
  def compareTo(self, other_graph):
    """
    Compute the Centrality measures in the graph.
    """
    print(f'\nComparing {self.name} to specified graph')
    own_measure_dicts = self.compute_graph_centrality_measures(self.G)
    other_measure_dicts = self.compute_graph_centrality_measures(other_graph) # cluster measures.
    measure_diffs = self.compare_measures(own_measure_dicts, other_measure_dicts)
    centrality_info = self.store_centrality_measures(self.df, measure_diffs, self.name)
    return centrality_info


class StoreNetworkMeasures:
  def __init__(self):
    self.G = None
    self.name = None
    self.df = None

  def load_graph(self, G, name):
    self.G =G
    self.name = name
    return self

  def compute_network_measures(self, G):
    # Network Density
    density = nx.density(G)
    print(f'Density of network: {density}')
    # Triadic closure property
    triadic_closure = nx.transitivity(G)
    print(f'Triadic Closure of network: {triadic_closure}')

    # Connectedness and components of the network
    print(f'Network is connected: {nx.is_strongly_connected(G)}')
    components = nx.strongly_connected_components(G)
    num_components = nx.number_strongly_connected_components(G)
    print(f'Number of components in the network: {num_components}')
    largest_component = max(components, key=len)
    subgraph = G.subgraph(largest_component)
    max_component_diameter = nx.diameter(subgraph)
    print(f'Number of nodes in the largest network: {len(subgraph)}')
    print("Network diameter of largest component:", max_component_diameter)
    return [density, triadic_closure, num_components, len(subgraph), max_component_diameter]

  def store_network_measures(self, df, measures, col_name):
    if self.df is None:
      network_measures = ['density', 'triadic_closure', 'num_components', 'max_component_len', 'max_component_diameter']
      centrality_measures = ['degree','closeness','betweenness','eigenvector']
      self.df = pd.DataFrame(network_measures, columns = ['index']).set_index('index')
    self.df[col_name] = [round(stats,3) for stats in measures]
    return self.df

  def save_csv(self, name):
    self.df.to_csv(name, index=False)

  @timer
  def run(self):
    print(f'\nStoring measures for {self.name}')
    network_measures = self.compute_network_measures(self.G)
    network_measure_info = self.store_network_measures(self.df, network_measures, self.name)
    return network_measure_info



########################################################################################################################


class network:
   def __init__(self, data):
     self.data = data
   def get_clusters(self):
     """
     Append the feature 'cluster" into the dataset
     """
     pass
  
   def prepare_NA_data(self, data):
      """
      Return the data used to create the networkx graph
      Should contain the source node, the target node, as well as the edge
      """
      pass
    
   def create_data(self):
      """
      After preparing the data for NA, we create the graph
      """
      pass
    
   def create_NA_features(self):
      """
      Create the features to be used for the dataset.
      Maybe to add in the centrality measures to each node as features
      """
      pass
    
   def compare_network(self):
      """
      Compare the statistical measures between the clusters
      Original Density vs Cluster1 up to CN
      Original Number of Components vs that in Clusters 1 up to N
      Original Triadic Closure vs that in Clusters 1 to N
      """

      pass

   def compare_nodes(self):
      """
      Original Degree vs that in Clusters 1 to N 
      """
      pass
      