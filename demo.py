import argparse
import glob
from peddla import peddla_net
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch

def parse_args():

    parser = argparse.ArgumentParser(description='Train SiamAF')
    parser.add_argument('--img_list', type=str, default='files of image list')

    args = parser.parse_args()
    return args

def preprocess(image, mean, std):
    img = (image - mean) / std
    return torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis, ...])

def parse_det(hm, wh, reg, density=None, diversity=None, score=0.1,down=4):
    # hm = _nms(hm, kernel=2)
    seman = hm[0, 0].cpu().numpy()
    height = wh[0, 0].cpu().numpy()
    offset_y = reg[0, 0, :, :].cpu().numpy()
    offset_x = reg[0, 1, :, :].cpu().numpy()
    density = density[0, 0].cpu().numpy()
    diversity = diversity[0].cpu().numpy()
    y_c, x_c = np.where(seman > score)
    maxh = int(down * seman.shape[0])
    maxw = int(down * seman.shape[1])
    boxs = []
    dens = []
    divers = []
    if len(y_c) > 0:
        for i in range(len(y_c)):
            h = np.exp(height[y_c[i], x_c[i]]) * down
            w = 0.41 * h
            o_y = offset_y[y_c[i], x_c[i]]
            o_x = offset_x[y_c[i], x_c[i]]
            s = seman[y_c[i], x_c[i]]
            x1, y1 = max(0, (x_c[i] + o_x) * down - w / 2), max(0, (y_c[i] + o_y) * down - h / 2)
            boxs.append([x1, y1, min(x1 + w, maxw), min(y1 + h, maxh), s])
            dens.append(density[y_c[i], x_c[i]])
            divers.append(diversity[:, y_c[i], x_c[i]])
        boxs = np.asarray(boxs, dtype=np.float32)
        dens = np.asarray(dens, dtype=np.float32)
        divers = np.asarray(divers, dtype=np.float32)
        keep = a_nms(boxs, 0.5, dens, divers)
        boxs = boxs[keep, :]
    else:
        boxs = np.asarray(boxs, dtype=np.float32)
    return boxs

def a_nms(dets, thresh, density, diversity):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        thresh_update = min(max(thresh, density[i]), 0.75)

        temp_tag = diversity[i]
        temp_tags = diversity[order[1:]]
        diff = np.sqrt(np.power((temp_tag - temp_tags), 2).sum(1))
        Flag_4 = diff > 0.95

        thresh_ = np.ones_like(ovr) * 0.5
        thresh_[Flag_4] = thresh_update
        inds = np.where(ovr <= thresh_)[0]
        order = order[inds + 1]

    return keep


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    return model

def main():
    # BGR
    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(1, 1, 3)

    args = parse_args()
    num_layers = 34
    heads = {'hm': 1, 'wh': 1, 'reg': 2, 'aed': 4}
    model = peddla_net(num_layers, heads, head_conv=256, down_ratio=4).cuda().eval()

    # load model
    model = load_model(model, 'final.pth')
    # torch.cuda.empty_cache()

    file_lists = sorted(glob.glob(args.img_list))
    for file in file_lists:
        torch.cuda.synchronize()
        img = plt.imread(file).astype(np.float32)
        img_pre = preprocess(img[:, :, ::-1], mean, std)
        img_pre = img_pre.cuda()

        with torch.no_grad():
            output = model(img_pre)[-1]
        output['hm'].sigmoid_()
        hm, wh, reg, attr = output['hm'], output['wh'], output['reg'], output['aed']

        density = attr.pow(2).sum(dim=1, keepdim=True).sqrt()
        diversity = torch.div(attr, density)
        boxes = parse_det(hm, wh, reg, density=density, diversity=diversity, score=0.5, down=4)

        if len(boxes) > 0:
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]

            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for i in range(len(boxes)):
                x, y, w, h, score = boxes[i]
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        else:
            plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    main()