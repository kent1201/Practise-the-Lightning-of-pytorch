import os
import io
import cv2
import numpy as np
# from rich.progress import track
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance
from utils import CheckSavePath, GetConfigs, GetNowTime_yyyymmddhhMMss
import gc

def CalEuclideanDistance(feature1: np.array, feature2: np.array):
    return distance.euclidean(feature1, feature2)

def CalCosineDistance(feature1: np.array, feature2: np.array):
    return distance.cosine(feature1, feature2)

def CompareFeature(becompared_item, compare_item, comapre_method="euclidean"):
    
    compare_image_path = compare_item["image_path"][0]
    compare_cams_image_path = compare_item["cams_image_path"][0]
    comapre_predict = compare_item["predict"]
    compare_label = compare_item["label"][0]
    compare_feature = compare_item["feature"]


    becompared_image_path = becompared_item["image_path"][0]
    becompared_cams_image_path = becompared_item["cams_image_path"][0]
    becompared_version = becompared_item["version"]
    becompared_predict = becompared_item["predict"]
    becompared_confidence = becompared_item["confidence"]
    becompared_site_predict = becompared_item["site-predict"]
    becompared_label = becompared_item["label"][0]
    becompared_feature = becompared_item["feature"]
    
    score = 0.0
    if comapre_method == "euclidean":
        score = CalEuclideanDistance(np.squeeze(becompared_feature, axis=0), np.squeeze(compare_feature, axis=0))
    elif comapre_method == "cosine":
        score = CalCosineDistance(np.squeeze(becompared_feature, axis=0), np.squeeze(compare_feature, axis=0))
    else:
        raise ValueError("No compare method {}".format(comapre_method))
    
    return {"becompared_image_path": becompared_image_path, 
            "becompared_cams_image_path": becompared_cams_image_path, 
            "becompared_version": becompared_version,
            "becompared_predict": becompared_predict,
            "becompared_confidence": becompared_confidence,
            "becompared_site_predict": becompared_site_predict,
            "becompared_label": becompared_label,
            "compare_image_path": compare_image_path, 
            "compare_cams_image_path": compare_cams_image_path, 
            "comapre_predict": comapre_predict,
            "compare_label": compare_label, 
            "distance": score}

def get_resized_image_data(file_path=None, image=None, cell_image_shape = (230,180, 3)): 
    try:
        if file_path:
            image = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.any():
            image = image

        max_shape = max(image.shape[0:2])
        new_image = np.zeros((max_shape,max_shape,3),np.uint8)
        ax,ay = (max_shape - image.shape[1])//2,(max_shape - image.shape[0])//2
        new_image[ay:image.shape[0]+ay,ax:ax+image.shape[1]] = image
        image = new_image
        image = cv2.resize(image,(cell_image_shape[0],cell_image_shape[1]))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        #res, im_JPEG = cv2.imencode('.png', image, imparams)
        res, im_JPEG = cv2.imencode('.jpg', image)
        byte_stream = io.BytesIO(im_JPEG)  
    except Exception as ex:
        print("error_image_file= %s, %s", file_path, ex)
        # fp.write(file_path)
        # fp.write("\n")
        raise ValueError('error_image_file', file_path, ex)
    return byte_stream
            


if __name__=='__main__':

    configs = GetConfigs(r"D:\Users\KentTsai\Documents\ViT_pytorch\K2\ModelB.ini")
    root_path = configs.get("03_xlsx", "root_path")
    becompared_mode = configs.get("03_xlsx", "becompared_mode")
    comapre_mode = configs.get("03_xlsx", "comapre_mode")
    top_k = configs.getint("03_xlsx", "top_k")
    distance_method = configs.get("03_xlsx", "distance_method")

    
    becompared_filename = os.path.join(root_path, "{}_features.npy".format(becompared_mode))
    compare_filename = os.path.join(root_path, "{}_features.npy".format(comapre_mode))
    save_dir =  CheckSavePath(root_path)
    # Set excel name
    save_xlsx_name = "{}_{}_{}_distance.xlsx".format(GetNowTime_yyyymmddhhMMss(), becompared_mode, distance_method)
    
    
    # label_dict = {0: "CP00", 1:"CP03", 2:"CP06" , 3:"CP08", 4:"CP09", 5:"DR02", 6:"IT03", 7:"IT08", 8:"IT09", 9:"Unknown"}
    compare_features = np.load(compare_filename, allow_pickle=True).tolist()
    becompared_features = np.load(becompared_filename, allow_pickle=True).tolist()

    
    # Create excel writer and dataframe
    save_path = os.path.join(save_dir, save_xlsx_name)
    writer = pd.ExcelWriter(save_path, engine='xlsxwriter')
    df_excel = pd.DataFrame(columns=['becompared_image', 'becompared_CAMs', 'becompared_image_path', 'becompared_version', 'becompared_label', 'becompared_predict', 'becompared_confidence', 'becomapred_site_predict', 
                                     'top_num', 'comapre_image', 'compare_CAMs', 'compare_image_path', 'compare_label', 'compare_predict', 
                                     'distance', "predict_same (becompared label and predict)", "label_same (becompared and compare label)", "predict_same (compare label and predict)", "predict_same (predict and site-predict)"])
    df_excel.to_excel(writer, index=False, sheet_name="Sheet1")
    # Get the xlsxwriter objects from the dataframe writer object.
    workbook  = writer.book
    worksheet = writer.sheets['Sheet1']
    worksheet.set_column('A:S', 35)
    worksheet.set_row(1, 150)
    write_row_index = 2


    comapre_results = list()
    for becompared_item in tqdm(becompared_features):
        
        # Compare each test feature to all train features
        comapre_results = list()
        for compare_item in compare_features:
            comapre_result = CompareFeature(becompared_item, compare_item, comapre_method=distance_method)
            comapre_results.append(comapre_result)
        df = pd.DataFrame(comapre_results)
        df.sort_values(by=["distance"], ascending=True, inplace=True)

        ## only save top-k results
        df_top_k = df.iloc[:top_k, :]
        
        del df
        gc.collect()

        flag_save_becompared_img = True

        ## Write results and images to excel
        top_num = 1
        for index, row in df_top_k.iterrows():
            
            worksheet.set_row(write_row_index, 150) # Row 1 height set to 20
            
            if flag_save_becompared_img:
                becomapred_image = get_resized_image_data(file_path=row['becompared_image_path'])
                worksheet.insert_image('A' + str(write_row_index), row['becompared_image_path'], {'x_offset': 5,'y_offset': 5,'object_position': 1,'image_data': becomapred_image})
                ## Get Predict and CAMs image
                becomapred_cams_image = get_resized_image_data(file_path=row['becompared_cams_image_path'])
                worksheet.insert_image('B' + str(write_row_index), row['becompared_cams_image_path'], {'x_offset': 5,'y_offset': 5,'object_position': 1,'image_data': becomapred_cams_image})
                worksheet.write_url('C' + str(write_row_index), row['becompared_image_path'])
                flag_save_becompared_img = False
                
            worksheet.write('D' + str(write_row_index), row['becompared_version'])
            worksheet.write('E' + str(write_row_index), row['becompared_label'])
            worksheet.write('F' + str(write_row_index), row['becompared_predict'])
            worksheet.write('G' + str(write_row_index), row['becompared_confidence'])
            worksheet.write('H' + str(write_row_index), row['becompared_site_predict'])
            
            comapre_image = get_resized_image_data(file_path=row['compare_image_path'])
            ## Get Predict and CAMs image
            comapre_cams_image = get_resized_image_data(file_path=row['compare_cams_image_path'])


            worksheet.write('I' + str(write_row_index), str(top_num))
            worksheet.insert_image('J' + str(write_row_index), row['compare_image_path'], {'x_offset': 5,'y_offset': 5,'object_position': 1,'image_data': comapre_image})
            worksheet.insert_image('K' + str(write_row_index), row['compare_cams_image_path'], {'x_offset': 5,'y_offset': 5,'object_position': 1,'image_data': comapre_cams_image})
            worksheet.write_url('L' + str(write_row_index), row['compare_image_path'])
            worksheet.write('M' + str(write_row_index), row['compare_label'])
            worksheet.write('N' + str(write_row_index), row['comapre_predict'])
            worksheet.write('O' + str(write_row_index), row['distance'])
            if row['becompared_label'] == row['becompared_predict']:
                worksheet.write('P' + str(write_row_index), "1")
            else:
                worksheet.write('P' + str(write_row_index), "0")
            if row['becompared_label'] == row['compare_label']:
                worksheet.write('Q' + str(write_row_index), "1")
            else:
                worksheet.write('Q' + str(write_row_index), "0")
            if row['compare_label'] == row['comapre_predict']:
                worksheet.write('R' + str(write_row_index), "1")
            else:
                worksheet.write('R' + str(write_row_index), "0")
            if row['becompared_predict'] == row['becompared_site_predict']:
                worksheet.write('S' + str(write_row_index), "1")
            else:
                worksheet.write('S' + str(write_row_index), "0")
            
            write_row_index += 1
            top_num += 1
        
    # worksheet.autofilter('A1:CC1')
    # workbook.close()
    # writer.save()
    writer.close()