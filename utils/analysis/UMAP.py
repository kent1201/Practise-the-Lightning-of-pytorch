import umap.umap_ as umap
import numpy as np
import configparser
import os

class UMAP(object):
    def __init__(self):

        fp_dir = os.getcwd()
        config_dir = os.path.join(fp_dir, 'utils{}analysis'.format(os.sep))
        config_path = os.path.join(config_dir, 'AnalysisMethodsConfig.ini')
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='UTF-8')

        n_neighbors = self.config.getint('UMAP', 'n_neighbors')
        min_dist = self.config.getfloat('UMAP', 'min_dist')
        n_components = self.config.getint('UMAP', 'n_components')
        metric = self.config.get('UMAP', 'metric')

        
        self.umap = umap.UMAP(n_neighbors=n_neighbors, 
                        min_dist=min_dist, 
                        n_components=n_components, 
                        metric=metric)

    def __call__(self, data_X):
        # TSNE anlaysis
        umap_results = self.umap.fit_transform(data_X)
        return umap_results