from sklearn.manifold import TSNE
import numpy as np
import configparser
import os

class tSNE(object):
    def __init__(self):

        fp_dir = os.getcwd()
        config_dir = os.path.join(fp_dir, 'utils{}analysis'.format(os.sep))
        config_path = os.path.join(config_dir, 'AnalysisMethodsConfig.ini')
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='UTF-8')

        n_components = self.config.getint('tSNE', 'n_components')
        early_exaggeration = self.config.getint('tSNE', 'early_exaggeration')
        learning_rate = self.config.getfloat('tSNE', 'learning_rate')
        min_grad_norm = self.config.getfloat('tSNE', 'min_grad_norm')
        verbose = self.config.getint('tSNE', 'verbose')
        perplexity = self.config.getint('tSNE', 'perplexity')
        n_iter = self.config.getint('tSNE', 'n_iter')
        random_state = self.config.getint('tSNE', 'random_state')
        init = self.config.get('tSNE', 'init')
        method = self.config.get('tSNE', 'method')
        angle = self.config.getfloat('tSNE', 'angle')
        
        self.tsne = TSNE(n_components=n_components, 
                        early_exaggeration=early_exaggeration,
                        learning_rate=learning_rate,
                        min_grad_norm=min_grad_norm,
                        verbose=verbose, 
                        perplexity=perplexity, 
                        n_iter=n_iter,
                        random_state=random_state, 
                        init=init, 
                        method=method, 
                        angle=angle)

    def __call__(self, data_X):
        # TSNE anlaysis
        tsne_results = self.tsne.fit_transform(data_X)
        return tsne_results