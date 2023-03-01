from PIL import  Image
import numpy as np
import cv2
from .keys import alphabetChinese as alphabet
# from keys import alphabetChinese as alphabet

import onnxruntime as rt
# from util import strLabelConverter, resizeNormalize
from .util import strLabelConverter, resizeNormalize

converter = strLabelConverter(''.join(alphabet))

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


class CRNNHandle:
    def __init__(self, model_path):

        self.sess = rt.InferenceSession(model_path)

    def predict(self, image):
        """
        预测
        """
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        transformer = resizeNormalize((w, 32))

        image = transformer(image)

        image = image.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)

        preds = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})

        preds = preds[0]


        length  = preds.shape[0]
        preds = preds.reshape(length,-1)

        preds = np.argmax(preds,axis=1)

        preds = preds.reshape(-1)


        sim_pred = converter.decode(preds, length, raw=False)

        return sim_pred



    def predict_rbg(self, crop_im, bbox):
        """
        预测
        """
        scale = crop_im.size[1] * 1.0 / 32
        w = crop_im.size[0] / scale
        w = int(w)


        img = crop_im.resize((w, 32), Image.BILINEAR)
        
        print('图片原始size [w, h]', crop_im.size, 'scale后的size [w, h]', img.size)
        img = np.array(img, dtype=np.float32)
        img -= 127.5
        img /= 127.5
        image = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)

        preds = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})

        preds = preds[0]

        print('crnn模型输出shape', preds.shape, f'每个输出代表的像素大小 {w / preds.shape[0] : .2f}',  )
        length  = preds.shape[0]
        preds = preds.reshape(length,-1)

        # preds = softmax(preds)


        preds = np.argmax(preds,axis=1)

        preds = preds.reshape(-1)

        result = converter.decode(preds, length, raw=True)

        # for item in result:
        raw, sim_pred, char_list = result['raw'], result['sim_pred'], result['char_list']

        for char_item in char_list:  # 添加原始坐标
            crop_left = char_item['idx'] * 4 * scale  # 在裁剪中的坐标
            crop_width = (char_item['idx'] + 1) * 4 * scale
            raw_left = bbox[0][0] + crop_left # 先简单的加上偏移
            raw_right = raw_left + crop_width

            char_item['bbox'] = [raw_left, bbox[0][1], raw_right, bbox[0][1] + crop_im.size[1]]

        return result



if __name__ == "__main__":
    im = Image.open("471594277244_.pic.jpg")
    crnn_handle = CRNNHandle(model_path="../models/crnn_lite_lstm_bk.onnx")
    print(crnn_handle.predict(im))