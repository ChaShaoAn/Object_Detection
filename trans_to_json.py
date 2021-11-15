import os
import cv2
import json
from tqdm import tqdm
import time

txtpath = 'runs/detect/exp/labels/'
testpath = 'data/svhn/test/'


def trans_to_json(txtpath='runs/detect/exp/labels/',
                  testpath='data/svhn/test/'):
    # Use the results from your model to generate the output json file
    # data_listdir = os.listdir("data/svhn/test/")
    data_listdir = os.listdir(testpath)
    data_listdir.sort(key=lambda x: int(x[:-4]))
    result_to_json = []

    # for each test image
    for img_name in tqdm(data_listdir):
        # the image_name is as same as the image_id
        image_id = int(img_name[:-4])

        im = cv2.imread(testpath + img_name)
        h, w, c = im.shape
        if not os.path.isfile(txtpath + img_name[:-4] + '.txt'):
            a = {
                "image_id": image_id,
                "bbox": [1, 1, 1, 1],
                "score": 0.5,
                "category_id": 0
            }
            result_to_json.append(a)
        else:
            f = open(txtpath + img_name[:-4] + '.txt')
            contents = f.readlines()
            for content in contents:
                a = {"image_id": 0, "bbox": [], "score": 0, "category_id": 0}
                content = content.replace('\n', '')
                c = content.split(' ')
                # print(c)
                # w_center = w*float(c[1])
                # h_center = h*float(c[2])
                x_left = w * float(c[1])
                y_top = h * float(c[2])
                width = w * float(c[3])
                height = h * float(c[4])
                x_left = x_left - (width / 2)
                y_top = y_top - (height / 2)
                a["image_id"] = image_id
                # a["bbox"] = (tuple([w_center, h_center, width, height]))
                a["bbox"] = (tuple([x_left, y_top, width, height]))
                a["score"] = (float(c[5]))
                a["category_id"] = (int(c[0]))

                result_to_json.append(a)
            f.close()

    # Write the list to answer.json
    json_object = json.dumps(result_to_json, indent=4)

    with open("answer.json", "w") as outfile:
        outfile.write(json_object)


trans_to_json(txtpath, testpath)
