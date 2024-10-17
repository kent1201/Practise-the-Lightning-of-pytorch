import os
import cv2
import random
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from utils.analysis import tSNE, PCA
from utils.analysis.draw_cam import draw_CAM
from utils.utils import CheckSavePath


# methods_map = {'pca': PCA.PCA(), 'tsne': tSNE.TSNE(), 'umap': UMAP.UMAP()}

plot_colors = ['b', 'g', 'r', 'tab:blue', 'tab:orange', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:olive', 'tab:cyan']

def ClassColors(classes= ["A", "B", "C", "D", "E"]):
    global plot_colors
    class_colors = dict()
    classes_count = len(classes)
    sample_colors = random.sample(plot_colors, classes_count)
    for sample_color, class_name in zip(sample_colors, classes):
        class_colors[class_name] = sample_color
    return class_colors

def LossCurve(losses, name, save_root="./"):
    global plot_colors
    colors_list = plot_colors.copy()
    random.shuffle(colors_list)
    loss_iter = range(len(losses))

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(loss_iter, losses, colors_list[0], label=name)
    ax.set_ylim((np.min(losses)-0.5, np.max(losses)+1.5))
    ax.set_xlabel("{} iteration".format(len(losses)))
    ax.set_ylabel("Loss")
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.0)
    ax.set_title(name)

    # save image
    save_img_path = os.path.join(save_root, name+".png")
    plt.savefig(save_img_path)  # should before show method

    # show
    # plt.show()


class DataVisualization:
    def __init__(self, args, model, dataset):
        self.analysis = args.analysis_method
        self.save_root = args.save_ckpt_path
        self.args = args
        self.save_img_name = ""
        self.model = model
        self.dataset = dataset
        self.filename_list = list()

    def dataTransform(self):
        data_X = []
        data_Y = []
        data_loader = self.dataset.test_dataloader()
        for batch in data_loader:
            images = batch[0]
            features = self.model.forward_features(images).detach().cpu()
            labels = batch[1].detach().cpu()
            images_path = batch[2]
            for idx in range(len(list(labels))):
                data_X.append(torch.flatten(features[idx]).numpy())
                data_Y.append(torch.argmax(labels[idx]).numpy())
                self.filename_list.append(images_path[idx])
        data_X = np.asarray(data_X)
        data_Y = np.asarray(data_Y)
        return data_X, data_Y

    def DrawCAM(self):
        data_loader = self.dataset.test_dataloader()
        for idx, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            images = batch[0]
            images_path = batch[2]
            for index in range(len(list(images_path))):
                save_path = CheckSavePath(os.path.join(self.save_root, "CAMs"))
                save_path = os.path.join(save_path, os.path.basename(images_path[index]))
                draw_CAM(self.model, images[index], images_path[index], save_path=save_path)
    
    def __Draw2DImage(self, analysis_result_df, lim, label_name='label', x_name='x', y_name='y', title='Analysis Result', analysis_name="PCA"):
        
        # Plotting
        fig = plt.figure(figsize=(30,30))
        ax = fig.add_subplot(1, 1, 1)
        sns.scatterplot(x=x_name, y=y_name, hue=label_name, style=label_name, data=analysis_result_df, ax=ax,s=120)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect('equal')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.0)
        ax.set_title(title)
        self.save_img_name = "{}_{}.png".format("2D", analysis_name)
        save_image_path = os.path.join(self.save_root, self.save_img_name)
        plt.savefig(save_image_path)
        plt.show()
    
    def __Draw3DImage(self, analysis_result_df, lim, label_name='label', x_name='x', y_name='y', z_name='z', title='Analysis Result', analysis_name="PCA"):
        
        fig = plt.figure(figsize=(30,30))
        ax = fig.add_subplot(projection='3d')
        class_list = np.unique(analysis_result_df[label_name])
        class_colors = ClassColors(class_list.tolist())
        for name in class_list.tolist():
            lable_df = analysis_result_df[analysis_result_df[label_name]==name]
            ax.scatter(lable_df[x_name], lable_df[y_name], lable_df[z_name], alpha=0.8, c=class_colors[name], edgecolors='none', s=40, label=name)
        # for x, y, z, group in zip(analysis_result_df[x_name].values, analysis_result_df[y_name].values, analysis_result_df[z_name].values, analysis_result_df['label']):
        #     ax.scatter([x], [y], [z], alpha=0.8, c=class_colors[group], edgecolors='none', s=40, label=group)
        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)
        ax.set_zlabel(z_name)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)
        ax.set_title(title)
        ax.set_aspect('auto')
        ax.legend(class_list, bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.0)
        self.save_img_name = "{}_{}.png".format("3D", analysis_name)
        save_image_path = os.path.join(self.save_root, self.save_img_name)
        plt.savefig(save_image_path, bbox_inches='tight')
        plt.show()
    
    def visualization(self, data_X, data_Y, sampled_data=10000):
        import time
        """Using PCA or tSNE for data visualization.
        Args:
        - ori_data: original data
        - generated_data: generated synthetic data
        - analysis: tsne or pca
        """
        # Analysis sample size (for faster computation)
        sample_size = min(data_X.shape[0], sampled_data)
        print("[visualization] number of sample: {}".format(sample_size))
        time.sleep(3)
        subsample_idc = np.random.choice(data_X.shape[0], sample_size, replace=True)
        
        data_x = data_X[subsample_idc,:]
        
        # y = np.unique(data_Y)
        data_y = data_Y[subsample_idc]

        data_name = np.asarray(self.filename_list)
        data_name = data_name[subsample_idc]
        
        analysis_result = None
        lim = None
        analysis_result_df = None
        
        if self.analysis == 'all' or self.analysis == 'pca':
            pca = PCA.PCA()
            analysis_result = pca(data_x)
            lim = (analysis_result.min()-5, analysis_result.max()+5)
            if analysis_result.shape[1] == 2:
                analysis_result_df = pd.DataFrame({'x-pca': analysis_result[:,0], 'y-pca': analysis_result[:,1], 'label': data_y})
                self.__Draw2DImage(analysis_result_df, lim, x_name='x-pca', y_name='y-pca', title='PCA plot', analysis_name="PCA")
            if analysis_result.shape[1] == 3:
                analysis_result_df = pd.DataFrame({'x-pca': analysis_result[:,0], 'y-pca': analysis_result[:,1], 'z-pca': analysis_result[:, 2], 'label': data_y})
                self.__Draw3DImage(analysis_result_df, lim, x_name='x-pca', y_name='y-pca', z_name='z-pca', title='PCA 3D plot', analysis_name="PCA")

        if self.analysis == 'all' or self.analysis == 'tsne':
            tsne = tSNE.tSNE()
            analysis_result = tsne(data_x)
            lim = (analysis_result.min()-5, analysis_result.max()+5)
            if analysis_result.shape[1] == 2:
                analysis_result_df = pd.DataFrame({'x-tsne': analysis_result[:,0], 'y-tsne': analysis_result[:,1], 'label': data_y})
                self.__Draw2DImage(analysis_result_df, lim, x_name='x-tsne', y_name='y-tsne', title='tSNE plot', analysis_name="tSNE")
            if analysis_result.shape[1] == 3:
                analysis_result_df = pd.DataFrame({'x-tsne': analysis_result[:,0], 'y-tsne': analysis_result[:,1], 'z-tsne': analysis_result[:, 2], 'label': data_y})
                self.__Draw3DImage(analysis_result_df, lim, x_name='x-tsne', y_name='y-tsne', z_name='z-tsne', title='tSNE 3D plot', analysis_name="tSNE")
        
        # if self.analysis == 'all' or self.analysis == 'umap':
        #     umap = UMAP.UMAP()
        #     analysis_result = umap(data_x)
        #     lim = (analysis_result.min()-5, analysis_result.max()+5)
        #     if analysis_result.shape[1] == 2:
        #         analysis_result_df = pd.DataFrame({'x-umap': analysis_result[:,0], 'y-umap': analysis_result[:,1], 'label': data_y})
        #         self.__Draw2DImage(analysis_result_df, lim, x_name='x-umap', y_name='y-umap', title='UMAP plot', analysis_name="UMAP")
        #     if analysis_result.shape[1] == 3:
        #         analysis_result_df = pd.DataFrame({'x-umap': analysis_result[:,0], 'y-umap': analysis_result[:,1], 'z-umap': analysis_result[:, 2], 'label': data_y})
        #         self.__Draw3DImage(analysis_result_df, lim, x_name='x-umap', y_name='y-umap', z_name='z-umap', title='UMAP 3D plot', analysis_name="UMAP")
            
        if self.analysis == 'all' or self.analysis == 'kmeans':
            
            tsne = tSNE.tSNE()
            analysis_result = tsne(data_x)
            lim = (analysis_result.min()-5, analysis_result.max()+5)

            labels_num = np.unique(data_y).shape[0]
            kmeans_labels = cluster.KMeans(n_clusters=labels_num).fit_predict(analysis_result)
            
            # clusterer_df = pd.DataFrame({'name': data_name, 'cluster': clusterer.labels_, 'label': data_y, 'x-umap': analysis_result[:,0], 'y-umap': analysis_result[:,1], 'z-umap': analysis_result[:,2]})
            clusterer_df = pd.DataFrame({'name': data_name, 'cluster': kmeans_labels, 'label': data_y, 'x-cluster': analysis_result[:,0], 'y-cluster': analysis_result[:,1], 'z-cluster': analysis_result[:,2]})
            sort_clusterer_df = clusterer_df.sort_values(by='cluster')
            sort_clusterer_df.to_csv(os.path.join(self.save_root, 'cluster.csv'))
            
            re_score = (
                adjusted_rand_score(data_y.tolist(), kmeans_labels),
                adjusted_mutual_info_score(data_y.tolist(), kmeans_labels)
            )
            print("adjusted_rand_score: {}\tadjusted_mutual_info_score: {}".format(re_score[0], re_score[1]))

            self.__Draw3DImage(clusterer_df, lim, label_name='cluster', x_name='x-cluster', y_name='y-cluster', z_name='z-cluster', title='Cluster 3D plot', analysis_name="KMeans")


