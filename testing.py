# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import torch
from torch.autograd import Variable

from src import utils, alphabet, dataset
import src.net as crnn


# -------------------路径参数-------------------

model_path = './models/crnn_chinese_5529.pth'
img_path = './test_imgs/images.jpeg'
alphabet = alphabet.alphabetChinese

# -------------------加载模型-------------------

model = crnn.CRNN(32, 1, len(alphabet) + 1, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)

model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

model.eval()

# -------------------预处理-------------------

converter = utils.strLabelConverter(alphabet)
image = Image.open(img_path).convert("RGB")
image = image.convert('L')

scale = image.size[1] * 1.0 / 32
w = image.size[0] / scale
w = int(w)
transformer = dataset.resizeNormalize((w, 32))

image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

# -------------------获取结果-------------------

preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)
preds_size = Variable(torch.IntTensor([preds.size(0)]))

raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
