import csv
import os
import pickle
import warnings
import cv2
import imutils
import numpy as np
import tqdm
from imutils import perspective, contours
from scipy.spatial import distance as dist


# Crop hand out of original image
def one_hand(image_gray):
    thresh = cv2.threshold(image_gray, 0, 255,cv2.THRESH_OTSU)[1]
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda z: cv2.contourArea(z), reverse=True)[0]
    (x, y, w, h) = cv2.boundingRect(cnts)

    center = (int(x + w/2), int(y+ h/2))
    return zip((x, y, w, h)), center



def resize_image(img, size=(28,28)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w:
        return cv2.resize(img, size, cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

def image_to_hand(image_gray, label):
    label_2hand = [29, 31, 32, 34, 35, 36, 43, 44, 45, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58, 60, 61, 63]
    if label in label_2hand:
        bb, center = two_hand(image_gray)
        return bb, center, "2"
    else:
        bb, center = one_hand(image_gray)
        return bb, center, "1"


def encode_images(path_to_image, path_to_pkl=None):
    f = open(path_to_image, 'r')
    list= f.readlines()
    list = list[ int(len(list)/3):2* int(len(list)/3)]
    video = []
    name = []

    videos = []
    labels = []
    f = open(path_to_pkl, 'wb')
    for (i, path) in enumerate(list):
        # print("[INFO] processing image {}/{}".format(i + 1, len(list)))

        path = path.split('\n')[0]
        label = path.split('/')[-1]

        label = int(label.split('_')[0])
        image = cv2.imread(path, 1)
        # image = cv2.merge((image, image, image))
        # image = np.expand_dims(image, -1)

        # name += [label]
        video += [image]
        if (i+1) % 30 == 0:
            video = np.array(video)
            label = np.array(label)

            videos += [video]
            labels += [label]

            # video = []
            # name = []

    videos = np.array(videos)
    labels = np.array(labels)
    print(videos.shape, labels.shape)
    data = {'labels': labels, 'videos': videos}
    f.write(pickle.dumps(data))
    f.close()




encode_images('/home/quan/PycharmProjects/SLR/Model/filename.txt', '/home/quan/PycharmProjects/SLR/Model/data_2.pkl')


#%%

data = pickle.loads(open('/home/quan/PycharmProjects/SLR/Model/data.pkl', 'rb').read())

vid = data['labels']

#%%
vid[2]


#%%
def frame_to_csv(path_to_frame, path_to_csv):
    # lists = os.listdir(path_to_frame)
    # print(lists)
    f = open(path_to_frame, 'r')
    lists = f.readlines()
    lists = lists[1:]
    with open(path_to_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        for image in tqdm.tqdm(lists):
            image = image.split('\n')[0]
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image = image.split('/')[-1]
            label = int(image.split('_')[0])
            value = np.asarray(img, dtype='uint8').reshape((img.shape[1], img.shape[0]))
            value = value.flatten()
            data = np.append(int(label), value)
            writer.writerow(data)




def two_hand(gray):
    def midpoint(ptA, ptB):
        return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

    blur = cv2.fastNlMeansDenoising(gray, None, 7, 5, 21)
    (T, thresh) = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=lambda z: cv2.contourArea(z), reverse=True)[:2]

    center = None
    coors = []
    refObj = None
    for c in cnts:
        # compute bouding box for the contour then extract the digit
        (x, y, w, h) = cv2.boundingRect(c)
        coor = [x, y, w, h]
        coors.append(coor)
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)

        box = np.array(box, dtype="int")
        box = perspective.order_points(box)
        cX = np.average(box[:, 0])
        cY = np.average(box[:, 1])
        if refObj is None:

            refObj = (box, (cX, cY))

            continue
        center = ((cX, cY), (refObj[1][0], refObj[1][1]))





    return coors, center







def stack_optical_flow(frames, mean_sub=False):
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
        warnings.warn('Warning! The data type has been changed to np.float32 for graylevel conversion...')
    frame_shape = frames.shape[1:-1]  # e.g. frames.shape is (10, 216, 216, 3)
    num_sequences = frames.shape[0]
    output_shape = frame_shape + (2 * (num_sequences - 1),)  # stacked_optical_flow.shape is (216, 216, 18)
    flows = np.ndarray(shape=output_shape)

    for i in range(num_sequences - 1):
        prev_frame = frames[i]
        next_frame = frames[i + 1]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = _calc_optical_flow(prev_gray, next_gray)
        flows[:, :, 2 * i:2 * i + 2] = flow

    if mean_sub:
        flows_x = flows[:, :, 0:2 * (num_sequences - 1):2]
        flows_y = flows[:, :, 1:2 * (num_sequences - 1):2]
        mean_x = np.mean(flows_x, axis=2)
        mean_y = np.mean(flows_y, axis=2)
        for i in range(2 * (num_sequences - 1)):
            flows[:, :, i] = flows[:, :, i] - mean_x if i % 2 == 0 else flows[:, :, i] - mean_y

    return flows


def _calc_optical_flow(prev, next_):
    flow = cv2.calcOpticalFlowFarneback(prev, next_, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    return flow




#%%

# path_train = '/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/filename_train.txt'
# path_test = '/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/filename_test.txt'
# Csv = '/home/quan/PycharmProjects/sign-language-gesture-recognition/data/train_sign.csv'
#
# import handsegment as hs
# count = 0
# cap = cv2.VideoCapture('/home/quan/PycharmProjects/sign-language-gesture-recognition/all/043_002_004.mp4')
# while True:
#     success, img1 = cap.read()
#     if success is False:
#         break
#     img = hs.handsegment(img1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     a, b = two_hand(img)
#     if len(a) < 2:
#         continue
#     x1, y1, w1, h1 = a[0][0], a[0][1], a[0][2], a[0][3]
#     x2, y2, w2, h2 = a[1][0], a[1][1], a[1][2], a[1][3]
#     h1 = img[y1 - 10:y1 + h1 + 10, x1 - 10:x1 + w1 + 10]
#     h2 = img[y2 - 10:y2 + h2 + 10, x2 - 10:x2 + w2 + 10]
#     count += 1
#     print(count)
#     h1 = resize_image(h1, (96, 96))
#     h2 = resize_image(h2, (96, 96))
#
#     hand = np.hstack([h1, h2])
#     cv2.imshow('i', hand)
#     cv2.waitKey(25)
#%%

# import os
# import numpy as np
#
# f1 = open('/Model/filename.txt', 'r')
# l = f1.readlines()
#
# var = l[72720]
#
# l = [os.path.splitext(i)[0] for i in l]
#
#
# l = np.asarray([int(i.split('_')[-1]) for i in l])
# #%%
# count = 0
# for i in range (3200):
#     for j in range(0, 30):
#         if j == l[count]:
#             print('yes')
#         else:
#             print(count)
#         count += 1
#
# #%%
# np.unique(l, return_counts=True)

