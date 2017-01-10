import numpy as np


def color_image(image, num_classes=20):
    import matplotlib as mpl
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))

def cal_IU(pred, truth, class_no):
    intersect = 0
    pred_cnt = 0
    truth_cnt = 0
    