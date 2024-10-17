import sklearn
import configparser
import os

class PCA(object):
    def __init__(self):

        fp_dir = os.getcwd()
        config_dir = os.path.join(fp_dir, 'utils{}analysis'.format(os.sep))
        config_path = os.path.join(config_dir, 'AnalysisMethodsConfig.ini')
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='UTF-8')

        n_components = self.config.getint('PCA', 'n_components')
        copy = self.config.getboolean('PCA', 'copy')
        whiten = self.config.getboolean('PCA', 'whiten')
        svd_solver = self.config.get('PCA', 'svd_solver')
        tol = self.config.getfloat('PCA', 'tol')
        iterated_power = self.config.get('PCA', 'iterated_power')
        random_state = self.config.getint('PCA', 'random_state')
        
        self.pca = sklearn.decomposition.PCA(n_components=n_components, 
                                                copy=copy, 
                                                whiten=whiten, 
                                                svd_solver=svd_solver, 
                                                tol=tol, 
                                                iterated_power=iterated_power,  
                                                random_state=random_state)

    def __call__(self, data_X):
        # TSNE anlaysis
        self.pca.fit(data_X)
        pca_result = self.pca.transform(data_X)
        return pca_result