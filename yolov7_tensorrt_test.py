from unittest import result
import cv2
import torch
import random
import time
import numpy as np
import tensorrt as trt
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
import argparse


class yoloV7_tensorrt():
    def __init__(self,weights_path):
        weights = weights_path
        device = torch.device('cuda:0')
        # Infer TensorRT Engine
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(weights, 'rb') as f, trt.Runtime(logger) as runtime:
            model = runtime.deserialize_cuda_engine(f.read())
        self.bindings = OrderedDict()
        for index in range(model.num_bindings):

            name = model.get_binding_name(index)
            dtype = trt.nptype(model.get_binding_dtype(index))
            shape = tuple(model.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = model.create_execution_context()

        self.names = [
            "Движение запрещено",
            "Скорость до 20км/ч",
            "Скорость до 30км/ч",
            "Скорость до 40км/ч",
            "Стоп",
        ]

        self.colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(self.names)}


    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)

        dw /= 2
        dh /= 2

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, r, (dw, dh)

    def postprocess(self,boxes,r,dwdh):
        dwdh = torch.tensor(dwdh*2).to(boxes.device)
        boxes -= dwdh
        boxes /= r
        return boxes

    

    def detect(self,img):
        device = torch.device('cuda:0')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = self.letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)

        im = torch.from_numpy(im).to(device)
        im/=255
        start = time.perf_counter()
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        nums = self.bindings['num_dets'].data
        boxes = self.bindings['det_boxes'].data
        scores = self.bindings['det_scores'].data
        classes = self.bindings['det_classes'].data

        boxes = boxes[0,:nums[0][0]]
        scores = scores[0,:nums[0][0]]
        classes = classes[0,:nums[0][0]]


        for box,score,cl in zip(boxes,scores,classes):
            box = self.postprocess(box,ratio,dwdh).round().int()
            name = self.names[cl]
            color = self.colors[name]
            name += ' ' + str(round(float(score),3))
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]
            cv2.rectangle(img,(xmin,ymin),(xmax,ymax),color,8)
            cv2.putText(img,name,(int(box[0]), int(box[1]) - 15),cv2.FONT_HERSHEY_SIMPLEX,2,color,thickness=5,lineType=cv2.LINE_AA)

        return img



def main(args):
    weights_path = args["weights"]
    image_path = args["image"]
    video_path = args["video"]
    out_path = args["output"]
    yolov7 = yoloV7_tensorrt(weights_path)


    if image_path is not None:
        image = cv2.imread(image_path)
        image = yolov7.detect(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if out_path is not None:
            result_name = "result.jpg"
            cv2.imwrite(out_path+result_name,image)
            
        cv2.imshow("image",cv2.resize(image,(1280,720)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weights", required=True, help="Путь к файлу результатов")
    ap.add_argument("-i", "--image", help="Путь к входному изображению")
    args = vars(ap.parse_args())
    main(args)