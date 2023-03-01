import onnxruntime as ort
ort.set_default_logger_severity(3)  # ERROR level 忽略 warning
from config import *
from crnn import CRNNHandle
from angnet import  AngleNetHandle
from utils import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import copy
from dbnet.dbnet_infer import DBNET
import time
import traceback

class  OcrHandle(object):
    def __init__(self):
        self.text_handle = DBNET(model_path)
        self.crnn_handle = CRNNHandle(crnn_model_path)
        if angle_detect:
            self.angle_handle = AngleNetHandle(angle_net_path)


    def crnnRecWithBox(self,im, boxes_list,score_list):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        boxes_list = sorted_boxes(np.array(boxes_list))

        line_imgs = []
        for index, (box, score) in enumerate(zip(boxes_list[:angle_detect_num], score_list[:angle_detect_num])):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))
            partImg = Image.fromarray(partImg_array).convert("RGB")
            line_imgs.append(partImg)

        angle_res = False
        if angle_detect:
            angle_res = self.angle_handle.predict_rbgs(line_imgs)

        count = 1
        for index, (box ,score) in enumerate(zip(boxes_list,score_list)):

            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))


            partImg = Image.fromarray(partImg_array).convert("RGB")

            if angle_detect and angle_res:
                partImg = partImg.rotate(180)


            if not is_rgb:
                partImg = partImg.convert('L')

            try:
                if is_rgb:
                    result = self.crnn_handle.predict_rbg(partImg, tmp_box)  ##识别的文本
                else:
                    simPred = self.crnn_handle.predict(partImg)  ##识别的文本
            except Exception as e:
                print(traceback.format_exc())
                continue

            # if simPred.strip() != '':
            #     results.append([tmp_box,"{}、 ".format(count)+  simPred,score])
            #     count += 1
            results.append({'bbox': tmp_box, 'result': result, 'score': score})

        return results


    def text_predict(self,img,short_size):
        boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8),short_size=short_size)
        result = self.crnnRecWithBox(np.array(img), boxes_list, score_list)

        return result


if __name__ == "__main__":
    ocrhandle = OcrHandle()
    short_size = 960
    # img = Image.open('invoice_001_000.png')
    img = Image.open('1.png')
    res = ocrhandle.text_predict(img,short_size)


    img_detected = img.copy()
    img_draw = ImageDraw.Draw(img_detected)
    colors = ['red', 'green', 'blue', "purple"]

    for i, r in enumerate(res):
        rect, txt, confidence = r['bbox'], r['result'], r['score']
        # print(txt)

        x1,y1,x2,y2,x3,y3,x4,y4 = rect.reshape(-1)
        size = max(min(x2-x1,y3-y2) // 2 , 15 )

        myfont = ImageFont.truetype("仿宋_GB2312.ttf", size=size)
        fillcolor = colors[i % len(colors)]
        # # img_draw.text((x1, y1 - size ), str(i+1), font=myfont, fill=fillcolor)
        # img_draw.text((x1, y1 - size ), str(txt), font=myfont, fill=fillcolor,)
        # for xy in [(x1, y1, x2, y2), (x2, y2, x3, y3 ), (x3 , y3 , x4, y4), (x4, y4, x1, y1)]:
        #     img_draw.line(xy=xy, fill=colors[i % len(colors)], width=2)

        for j, char_item in enumerate( txt['char_list']):
            char, (x1, y1, x2, y2) = char_item['char'], char_item['bbox']
            # img_draw.rectangle((x1, y1, x2, y2),  outline=colors[j % len(colors)], width=1)
            img_draw.text((x1, y1 - size ), str(char), font=myfont, fill=fillcolor,)
            img_draw.rectangle((x1, y1, x1, y2),  outline=fillcolor, width=1)

    img_detected.save('output1.jpg', format='JPEG')