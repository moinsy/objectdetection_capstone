import json
import pandas as pd


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    yA = max(boxA[0], boxB[0])
    xA = max(boxA[1], boxB[1])
    yB = min(boxA[2], boxB[2])
    xB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


with open('test_images_result.json') as f:
    test_j = json.load(f)

test_df = pd.read_csv('product_bbox.csv')

pdts = pd.read_csv('/home/justdial/PycharmProjects/objectdetection_oid/data/products.csv')


def get_label(id):
    return pdts[pdts['id'] == id].labelid.values[0]


def get_predicted_box(img):
    img = img + '.jpg'
    if img not in test_j:
        return None

    test_j[img]
    indx = [i for i, x in enumerate(test_j[img]['scores']) if x > 0.5]
    pred_res = {}
    for i, cls in enumerate(test_j[img]['classes'][:len(indx)]):
        pred_res[get_label(cls)] = test_j[img]['boxes'][i]

    return pred_res

def getmAP():
    res = {}
    tp,fp,fn,precision,recall,p_recall,ap = 0,0,0,0,0,0,0

    u_imgs = test_df.ImageID.unique().tolist()

    for i,img in enumerate(u_imgs):
    #     tp,fp,fn,precision,recall = 0,0,0,0,0
        annon = test_df[test_df.ImageID.isin([img])]
        pred_res = get_predicted_box(img)
        if not pred_res:
            continue
    #     print ("img:{}".format(img))
        for row in annon.iterrows():
            label = row[1]['LabelName']
            box = [row[1]['YMin'],row[1]['XMin'],row[1]['YMax'],row[1]['XMax']]
            label_check = False
            for key in pred_res:
                boxp = pred_res[key]
                iou = bb_intersection_over_union(box,boxp)
    #             print ("label:{},{}, iou:{}".format(label,key,iou))
                if iou > 0.5:
                    if label == key:
                        tp+=1
                    else:
                        fp+=1
                label_check = True
            if not label_check:
                fn+=1

        if tp+fp == 0 and tp+fn!=0:
            precision = 0
            recall = float(tp)/(tp+fn)

        elif tp+fn == 0 and tp+fp!=0:
            recall = 0
            precision = float(tp)/(tp+fp)

        else:
            precision = float(tp)/(tp+fp)
            recall = float(tp)/(tp+fn)

    #     print ("tp:{}, fp:{}, fn:{}".format(tp,fp,fn))


        rate_recall = recall-p_recall
    #     print ('r_r:{}'.format(rate_recall))

        ap += float(precision) * float(rate_recall)
        mAP = float(ap)/(i+1)
    #     res[img] = {'tp':tp, 'fp':fp, 'fn':fn, 'precision':precision, 'recall':recall, 'ap':ap, 'mAP':mAP}
    #     print res[img]
    print ('The mean Average Precision (mAP) for test images is : {}'.format(mAP))