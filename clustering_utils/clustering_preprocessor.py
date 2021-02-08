import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class bt4103Preprocessor:
    def __init__(self, data):
        self.data = data# data
        self.pca = None
        self.columns = data #data.columns
        
    def scale_numerical_data(self, data):
        self.data = data
        self.columns = data.columns
        sc = StandardScaler()
        sc.fit(data)
        transformed_data = sc.transform(data)
        self.transformed_data = transformed_data
        return self.transformed_data

    def perform_pca(self, n_components=None, seed=4103):
        
        pca = PCA(n_components=n_components, whiten=True, random_state=seed)
        self.data = self.scale_numerical_data(self.data)
        pca.fit(self.data)
        self.pca_data = pca.transform(self.data)
        self.pca = pca
        return self.pca_data

    def plot_pca_result(self, ax=None, n_components=None, seed=4103):
        if self.pca is None:
            self.perform_pca(n_components,seed)
            
        components = pd.DataFrame(
            np.round(self.pca.components_, 4), columns=self.columns)
        components.index = [f'D{i}' for i in range(
            1, len(self.pca.components_)+1)]
        ratios = self.pca.explained_variance_ratio_.reshape(-1, 1)
        variance_ratios = pd.DataFrame(
            np.round(ratios, 4), columns=['Explained Variance'])
        variance_ratios.index = components.index
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 8))
            # Plot the feature weights as a function of the components
            components.plot(ax=ax, kind='bar')
            ax.set_ylabel('Feature Weights')
            ax.set_xticklabels(components.index, rotation=0)
            for i, ev in enumerate(self.pca.explained_variance_ratio_):
                ax.text(i-0.40, ax.get_ylim()[1] + 0.05,
                           "%.4f" % (ev))
            plt.legend(title='Features', bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
        else:
            components.plot(ax=ax, kind='bar')
            ax.set_ylabel('Feature Weights')
            ax.set_xticklabels(components.index, rotation=0)
            for i, ev in enumerate(self.pca.explained_variance_ratio_):
                ax.text(i-0.40, ax.get_ylim()[1] + 0.05,
                           "%.4f" % (ev))
      
        
    def pca_result(self):
        components = pd.DataFrame(
            np.round(self.pca.components_, 4), columns=self.columns)
        components.index = [f'D{i}' for i in range(
            1, len(self.pca.components_)+1)]
        ratios = self.pca.explained_variance_ratio_.reshape(-1, 1)
        variance_ratios = pd.DataFrame(
            np.round(ratios, 4), columns=['Explained Variance'])
        variance_ratios.index = components.index

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
        # Plot the feature weights as a function of the components
        components.plot(ax=ax[0], kind='bar')
        ax[0].set_ylabel('Feature Weights')
        ax[0].set_xticklabels(components.index, rotation=0)

        for i, ev in enumerate(self.pca.explained_variance_ratio_):
            ax[0].text(i-0.40, ax[0].get_ylim()[1] + 0.05,
                       "%.4f" % (ev))

            ax[1].plot(np.cumsum(self.pca.explained_variance_ratio_))
            ax[1].set_xlabel('number of components')
            ax[1].set_ylabel('cumulative explained variance')
            ax[1].set_title('Cumulative Variance Ratio Plot')
            fig.tight_layout()
        return pd.concat([variance_ratios, components], axis=1)
