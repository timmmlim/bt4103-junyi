import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, Birch, AffinityPropagation, OPTICS
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
import seaborn as sns
from clustering_utils.clustering_preprocessor import bt4103Preprocessor
import functools
import time


def timer(func):
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


class bt4103Clustering:
    def __init__(self, data):
        self.unique_id = data.iloc[:, 0:1]
        self.data = data.iloc[:, 1:]
        self.inputs = None
        self.seed = 4103
        self.k_range = None
        self.n_trials = None
        self.clustering_scores = pd.DataFrame()
        self.clustering_groups = data.iloc[:, 0:1]
        self.max_score = -1

    def create_algo_input(self, inputs):
        self.inputs = inputs
        self.n_trials = len(inputs[0][2][1])
        return "Custom input has been created."

    def update_best_model(self, silhouette_scores, model_per_k):
        if max(silhouette_scores) > self.max_score:
            self.max_score = max(silhouette_scores)
            self.best_model = model_per_k[np.argmax(silhouette_scores)]
        return

    @timer
    def clustering_preprocessing(self, n_components=None):
        preprocessor = bt4103Preprocessor(self.data)
        self.data = preprocessor.scale_numerical_data()
        pca_data = preprocessor.perform_pca(n_components=n_components)
        preprocessor.pca_result()
        self.preprocessor = preprocessor
        self.data = preprocessor.pca_data
        return 

    def initialize_algo_inputs(self, k_range):
        """
        Initializes a DEFAULT list of of tuple (sklearn model, custom_parameters, hyperparameters for each of the k trials)
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html#sklearn.cluster.MeanShift
        # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html#sklearn.cluster.SpectralClustering
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html#sklearn.cluster.DBSCAN
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html#sklearn.cluster.OPTICS
        # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html#sklearn.cluster.Birch
        """
        if k_range is None:
            self.k_range = np.arange(2, 8)
        else:
            self.k_range = k_range
        self.n_trials = len(self.k_range)
        # tuple of the model and the corresponding model parameters, as well as the range of parameters that you want to try in the same number of trials
        self.inputs = [(KMeans, {'random_state': self.seed}, ('n_clusters', k_range)),
                      (GaussianMixture, {
                       'random_state': self.seed}, ('n_components', k_range)),
                      (MeanShift, {'bin_seeding': True},
                       ('bandwidth', k_range)),
                      (AgglomerativeClustering, {}, ('n_clusters', k_range)),
                      (DBSCAN, {'n_jobs': -1},
                       ('eps', np.linspace(0.2, 0.8, len(k_range)))),
                      (OPTICS, {'metric': 'cosine'}, (k_range,)),
                      (Birch, {}, ('n_clusters', k_range)),
                      # (AffinityPropagation, {'random_state':self.seed}, ()),
                      #  (SpectralClustering, {'random_state':self.seed}, ('n_clusters', k_range)),
                      ]

        return 'Default input initialized.'

    def update_params(self, params, hp, k):
        params[hp] = k
        return params

    def get_model_per_k(self, clustering_model, params, n_trials_hp):
        # if there is a parameter to update in each of the trials:
        if len(n_trials_hp) > 1:
            # for some reason need to cast the type to int for Birch
            if clustering_model.__name__ == "Birch":
                model_per_k = [clustering_model(**self.update_params(params, hparam, k)).fit(
                    self.data) for hparam, k in map(lambda x: (n_trials_hp[0], int(x)), n_trials_hp[1])]
            else:
                model_per_k = [clustering_model(**self.update_params(params, hparam, k)).fit(
                    self.data) for hparam, k in map(lambda x: (n_trials_hp[0], x), n_trials_hp[1])]
            return model_per_k
        # if there is NO parameter to update in each of the trails:
        else:
            model_per_k = [clustering_model(**params).fit(self.data)]
            return model_per_k

    def get_silhouette_list_scores(self, model_per_k):
        if model_per_k[0].__class__.__name__ == 'GaussianMixture':
            preds = [model.predict(self.data) for model in model_per_k]
            try:
                silhouette_scores = [silhouette_score(
                    self.data, model.predict(self.data)) for model in model_per_k]
            except ValueError:
                # cannot calculate sihouette score
                print(
                    f'{model_per_k.__class__.__name__} failed to find more than 1 cluster, thus terminated.\n')
                silhouette_scores = [0] * len(model_per_k)
        else:
            try:
                silhouette_scores = [silhouette_score(
                    self.data, model.labels_) for model in model_per_k]
            except ValueError:
                print(
                    f'{model_per_k.__class__.__name__} failed to find more than 1 cluster, thus terminated.\n')
                silhouette_scores = [0] * len(model_per_k)
        return silhouette_scores

    def get_best_model_labels(self, clustering_model, model_per_k, silhouette_scores, n_trials_hp):
        # if there is a parameter to update in each of the trials
        if len(n_trials_hp) > 1:
            best_k = n_trials_hp[1][np.argmax(silhouette_scores)]
            # If the clustering model is GaussianMixture, need to do .predict() to get back the labels as opposed to .labels_
            if clustering_model.__name__ == 'GaussianMixture':
                best_model_labels = model_per_k[int(
                    np.argmax(silhouette_scores))].predict(self.data)
            else:
                best_model_labels = model_per_k[int(
                    np.argmax(silhouette_scores))].labels_
            return best_k, best_model_labels
        # if there are no parameters to update in each of the trails
        else:
            best_k = -1  # there is no best parameter since we did not even update any hp in each trial
            if clustering_model.__name__ == 'GaussianMixture':
                best_model_labels = model_per_k[0].predict(self.data)
            else:
                best_model_labels = model_per_k[0].labels_
            # update this silhouette score as well because this score does not change in each number of trials
        return best_k, best_model_labels

    @timer
    def run_clustering(self, clustering_model, model_params, n_trials_hp):
        model_per_k = self.get_model_per_k(
            clustering_model, model_params, n_trials_hp)
        # if n_trials_hp does not affect the number of parameter, increase it to fit into the final resulting dataframe.
        silhouette_scores = self.get_silhouette_list_scores(model_per_k) if len(
            model_per_k) > 1 else self.get_silhouette_list_scores(model_per_k) * self.n_trials
        self.update_best_model(silhouette_scores, model_per_k)
        best_k, best_model_labels = self.get_best_model_labels(
            clustering_model, model_per_k, silhouette_scores, n_trials_hp)
        return silhouette_scores, (best_k, best_model_labels)

    def update_cluster_param(inputs, current_k):
        """
        Updates the 'n_cluster' parameter for sklearn.cluster models.
        input = [(KMeans, {'n_clusters':current_k}),
              (MeanShift, {'n_clusters':current_k}),
              ]
        """
        for model, param in inputs:
            param['n_clusters'] = current_k
        return inputs

    @timer
    def run_clustering_pipeline(self):
        new_clustering_scores = pd.DataFrame()
        new_clustering_groups = pd.DataFrame()
        for model, params, n_trials_hp in self.inputs:
            print(f'Beginning iteration for {model.__name__}')
            model_scores, best_model_res = self.run_clustering(
                model, params, n_trials_hp)

            new_clustering_scores[(model.__name__+'_hparams')] = list(map(lambda x: n_trials_hp[0] +
                                                                          '_' + str(round(x, 3)), n_trials_hp[1])) if len(n_trials_hp) > 1 else -1
            new_clustering_scores[(model.__name__)] = model_scores
            new_clustering_groups[(
                f'{model.__name__}_{n_trials_hp[0]}_{best_model_res[0]}')] = best_model_res[1]
            print(f'Finished iteration for {model.__name__}\n')
        self.clustering_scores = new_clustering_scores
        self.clustering_groups = pd.concat(
            [self.unique_id, new_clustering_groups], axis=1)
        return self.clustering_scores, self.clustering_groups

    def silhouette_analysis(self):
        fig, ax = plt.subplots(figsize=(16, 8))
        for models in map(lambda x: x[0], self.inputs):
            ax.plot(np.arange(len(self.clustering_scores[models.__name__+'_hparams'])),
                    self.clustering_scores[models.__name__], label=models.__name__)
        plt.xlabel('Number of Trials', fontsize=14)
        plt.ylabel('Sihouette score', fontsize=14)
        plt.legend()
        plt.show()
