檔案說明:
1. 00_CreateDataset.py: 從來源資料夾建立資料集 (建議自行處理)，輸出資料集格式參考資料集目錄結構
2. 01_DrawCAM.py: 轉換onnx至pth，並對特定資料夾({root_path}/{mode_type}/) 下符合 data_fmt 的圖像繪製出熱力圖並保存至 {root_path}/{mode_type}_CAMs/
3. 02_save_features.py: 會對 {root_path}/{mode_type}/ 下符合 data_fmt 的圖像進行特徵萃取並並保存至 {root_path}/{mode_type}_features.npy，內容還包含原始圖像路徑，類別，熱力圖像路徑，模型預測結果
4. 03_xlsx.py: 會將被比較的圖 (位於 {root_path}/{becompared_mode} 中)，每張圖會與位於 {root_path}/{comapre_mode} 中的所有圖一一進行比較，並將分析結果(.xlsx)輸出至 {root_path} 底下
模型目錄結構(以 SALA 平台提供的下載資料集為例，內容須包含: *json | *.ini | *.onnx)
--AI_Model\
    --Model_A\
        --*.json: 包含前處理設定
        --*.ini: 包含基於順序的類別名稱
        --*.onnx: onnx模型檔案
    --Model_B\
        --*.json: 包含前處理設定
        --*.ini: 包含基於順序的類別名稱
        --*.onnx: onnx模型檔案

資料集目錄結構:
model_type: 可為任意名稱, 例如 Train, Val, Test, ... 
--{root}\
    --{mode_type}\
        --label1\
            --001.jpg
            --002.jpg
            ...
            --N.jpg
        --label2\
            --001.jpg
            --002.jpg
            ...
            --N.jpg
        --labelM\
            --001.jpg
            --002.jpg
            ...
            --N.jpg

MDC.txt 檔案結構:
    file_path,label
    file_path,label
    file_path,label
    ...
    file_path,label
