# 使用 TensorFlow 的官方映像作為基礎映像
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

# 設置工作目錄，所有的工作都將在這個目錄下進行
WORKDIR /workspace

# （可選）安裝其他系統級依賴
# 這裡可以加入其他系統工具或依賴
RUN apt-get update && apt-get install -y git
# 更新 pip，並安裝其他必要的 Python 套件
RUN pip install --upgrade pip
RUN pip install numpy pandas matplotlib scikit-learn onnx==1.14.1 pillow scipy seaborn tensorboard timm lightning tqdm albumentations lion-pytorch opencv-python-headless
RUN pip install -U rich



# 暴露 JupyterLab 的端口（如果你需要使用 JupyterLab）
EXPOSE 8888

# 啟動命令，你可以選擇啟動 bash 進行開發，也可以啟動 JupyterLab
CMD ["bash"]

# 或者啟動 JupyterLab（可選）
# CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]