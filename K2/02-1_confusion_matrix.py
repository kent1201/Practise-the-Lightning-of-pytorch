import os
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from utils import GetConfigs, LabelStr2Dict, GetNowTime_yyyymmddhhMMss


def GetFeaturesInfo(features):
    info_list = list()
    for feature in features:
        image_path = feature["image_path"][0]
        predict = feature["predict"]
        label = feature["label"][0]
        # compare_feature = features["feature"]
        confidence = feature["confidence"]
        info_list.append({"image_path": image_path, 
                          "confidence": confidence, 
                          "predict": predict, 
                          "label": label, 
                          })
    return info_list

def NaN2Zero(matrix):
    where_are_NaNs = np.isnan(matrix)
    matrix[where_are_NaNs] = 0.0
    return matrix

def DrawConfusionMartrix(conf_matrix, conf_thres, x_class_name, y_class_name, save_name='./output.png', cmap='turbo', show=False):
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    # Using Seaborn heatmap to create the plot
    fmt="d"
    if conf_matrix.dtype==np.float64:
        fmt=".2f"
    elif conf_matrix.dtype==np.int64:
        fmt="d"
    else:
        fmt=""
    fx = sn.heatmap(conf_matrix, annot=True, cmap=cmap, fmt=fmt)

    # labels the title and x, y axis of plot
    fx.set_title('Confusion Matrix\n\n')
    fx.set_ylabel('Ground truth')
    fx.set_xlabel('Predicted Values') 

    fx.text(12, 6.7, "conf: {}".format(conf_thres), fontsize=9, fontweight="semibold", fontstyle="italic")

    # labels the boxes
    fx.xaxis.set_ticklabels(x_class_name)
    fx.yaxis.set_ticklabels(y_class_name)
    plt.savefig(save_name)
    if show:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()

class ConfusionMatrix:
    def __init__(self, label_dict: dict, CONF_THRESHOLD=0.0):
        self.label_dict = label_dict
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.num_classes = len(label_dict.values())
        self.matrix = np.zeros((self.num_classes, self.num_classes))

    def __GetKeybyValue(self, dict, search_value):
        for key, value in dict.items():
            if value == search_value:
                return key   

    def Calculate(self, predicts: np.ndarray, labels: np.ndarray) -> None:
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Arguments:
            predicts (Array[M, 2]) conf, class
            labels (Array[M, 1]), class
        Returns:
            None, updates confusion matrix accordingly
        """
        self.matrix = np.zeros((self.num_classes, self.num_classes))

        for predict, label in zip(predicts, labels):
                if float(predict[0]) >= self.CONF_THRESHOLD:
                    self.matrix[self.__GetKeybyValue(self.label_dict, label[0]), self.__GetKeybyValue(self.label_dict, predict[1])] += 1
                else:
                    self.matrix[self.__GetKeybyValue(self.label_dict, label[0]), len(self.label_dict.values())-1] += 1

    def return_matrix(self):
        return self.matrix
    
    def return_avg_matrix(self):
        np.seterr(divide='ignore',invalid='ignore')
        result_matrix = self.matrix / np.sum(self.matrix, axis = 0)
        where_are_NaNs = np.isnan(result_matrix)
        result_matrix[where_are_NaNs] = 0.0
        return result_matrix

    def print_matrix(self):
        for i in range(self.num_classes):
            print(' '.join(map(str, self.matrix[i])))

if __name__=='__main__':

    configs = GetConfigs(r"D:\Users\KentTsai\Documents\ViT_pytorch\K2\ModelB.ini")
    root_path = configs.get("02-1_confusion_matrix", "root_path")
    feature_path = configs.get("default", "mode_type")
    label_dict = LabelStr2Dict(configs.get("default", "labels"))
    confidence_threshold = configs.getfloat("02-1_confusion_matrix", "confidence_threshold")

    label_dict[len(label_dict.keys())] = "Unknown"

    save_csv_path = os.path.join(root_path, "{}_{}_matrix.csv".format(GetNowTime_yyyymmddhhMMss(), feature_path))
    save_img_path = os.path.join(root_path, "{}_{}_matrix.png".format(GetNowTime_yyyymmddhhMMss(), feature_path))

    classification_metrics = ConfusionMatrix(label_dict=label_dict, CONF_THRESHOLD=confidence_threshold)

    features = np.load("{}_features.npy".format(os.path.join(root_path, feature_path)), allow_pickle=True).tolist()

    info = GetFeaturesInfo(features)
    # for index, item in enumerate(info):
    #     pred = item["predict"]
    #     confidence = item["confidence"]
    #     label = item["label"]
    #     image_path = item["image_path"]
    #     print("{}: image_path: {}\tpred: {}\t conf: {}\tlabel: {}".format(index, image_path, pred, confidence, label))
    preds = np.asarray([[float(item["confidence"]), item["predict"]] for item in info])
    labels = np.asarray([[item["label"]] for item in info])
    # print(preds)

    classification_metrics.Calculate(preds, labels)

    number_matrix, avg_matrix = None, None
    try:
        ## Get confusion matrix as actual number 
        number_matrix = classification_metrics.return_matrix()
        ## Get confusion matrix as recall
        avg_matrix = np.asarray(classification_metrics.return_avg_matrix())
    except Exception as ex:
        raise RuntimeError("Get confusion matrix: {}".format(ex))
    
    ## Calculate precision and recall
    true_pos, precision, recall, avg_precision, avg_recall = None, None, None, None, None
    try:
        true_pos = np.diag(number_matrix)
        precision = true_pos / np.sum(number_matrix, axis=1)
        recall = true_pos / np.sum(number_matrix, axis=0)
        ## If labels is zero, division by zero will cause nan, change nan to zero
        precision = NaN2Zero(precision)
        recall = NaN2Zero(recall)
        avg_precision = np.average(precision)
        avg_recall = np.average(recall)
    except Exception as ex:
        raise RuntimeError("Calculate precision, recall error. {}".format(ex))
    
    ## Store results into .csv
    try:
        
        number_matrix_df = pd.DataFrame(number_matrix, index=label_dict.values(), columns=label_dict.values())
        number_matrix_df.loc['precision'] = precision.tolist()
        number_matrix_df.loc['recall'] = recall.tolist()
        number_matrix_df.loc['average precision'] = avg_precision.tolist()
        number_matrix_df.loc['average recall'] = avg_recall.tolist()
        number_matrix_df.to_csv(save_csv_path)
    except Exception as ex:
        raise RuntimeError("Save to csv error. {}".format(ex))
    

    ## Plot confusion matrix
    try:
        DrawConfusionMartrix(number_matrix.astype(np.int64), conf_thres=confidence_threshold, x_class_name=label_dict.values(), y_class_name=label_dict.values(), save_name=save_img_path, show=False)
    except Exception as ex:
        raise RuntimeError("Save to image error. {}".format(ex))

