# Object_Detection
This is deep learning homework2, The proposed challenge is a street view house numbers detection, which contains:
1. classify the digits of bounding boxes into 10 classes (0-9)
2. find left, top, width and height of bounding boxes which contain digits in a given image, and save them all to json file

### Environment
- Python 3.8.11
- Pytorch 1.9.1
- CUDA 11.1
- Yolo requirement
```
# install yolo requirements
pip install -r requirements.txt
```
## How to train
1. Download the given dataset from [Google Drive](https://drive.google.com/drive/folders/1aRWnNvirWHXXXpPPfcWlHQuzGJdXagoc)
2. make sure `train images` and `digitStruct.mat` is in `data/svhn/train`
3. cd to `data/svhn` and run `preprocess_mat_file.py`, it will seperate train and valid images to different folder (don't create valid folder by your own, or it will fail), and generate label in txt file for each iamge
```
# in data/svhn
python preprocess_mat_file.py
```
4. now you can begin training model (`yolov5m.pt` will auto download)
```
python train.py --img 320 --batch 16 --epochs 50 --data svhn.yaml --weights yolov5m.pt
```
### Inference and How to Reproducing Submission
- Use the Google Colab: [Inference.ipynb](https://colab.research.google.com/drive/1Cq_vvYy0fz8icnJDvjqlIRcaNPIy0VHW?usp=sharing)

### Reference
- [h5py - Quick Start Guide](https://docs.h5py.org/en/stable/quick.html)
- [YOLOv5](https://github.com/ultralytics/yolov5)