import cv2
import os
from os.path import join, exists
from utils import handsegment as hs
from tqdm import tqdm
import shutil

root = "/home/quan/PycharmProjects/sign-language-gesture-recognition/lsa64_hand_videos"
data_test_folder ="/home/quan/PycharmProjects/sign-language-gesture-recognition/Test_videos"
data_train_folder = "/home/quan/PycharmProjects/sign-language-gesture-recognition/Train_video"
def split_train_test_video(root_folder, data_test_folder, data_train_folder ):
    videos = os.listdir(root_folder)
    list = [14, 15, 17, 21]

    for video in videos:
        #split name of video with "_" and change into "int"
        label = int(video.split("_")[0])
        path_to_video = root_folder +"/" + video
        path_to_data = data_train_folder + "/{}".format(label)

        for i in list:
            if label == i:
                shutil.move(path_to_video, path_to_data)
                print("[INFO] Moving {} to {}".format(path_to_video, path_to_data))

    folders = os.listdir(data_train_folder)
    for (i,folder) in enumerate(folders):
        train_folder = data_train_folder + "/" + folder
        test_vid_folder = data_test_folder + "/" + folder

        videos_data = os.listdir(train_folder)

        for (j,vid) in enumerate(videos_data):
            if j < 10 :
                train_vid_path = train_folder + "/"+ vid
                shutil.move(train_vid_path, test_vid_folder)
                print("[INFO] Moving {} to {}".format(train_vid_path,test_vid_folder))


#split_train_test_video(root, data_test_folder, data_train_folder)

def convert(gesture_folder, target_folder):
    f = open('/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/filename.txt', 'a')
    rootPath = os.getcwd()
    target_folder = os.path.abspath(target_folder)
    #
    if not exists(target_folder):
        os.makedirs(target_folder)
    #
    gesture_folder = os.path.abspath(gesture_folder)

    os.chdir(gesture_folder)


    videos = os.listdir(gesture_folder)
    videos = videos[:1000]

    videos = [video for video in videos if (os.path.isfile(video))]
    for video in tqdm(videos, unit='videos', ascii=True):
        name = os.path.abspath(video)
        cap = cv2.VideoCapture(name)  # capturing input video
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lastFrame = None

        count = 0

        # assumption only first 150 frames are important
        while count < 150:
            ret, frame = cap.read()  # extract frame
            if ret is False:
                break
            framename = os.path.splitext(video)[0]
            framename = framename + "_frame_" + str(count) + ".png"

            framename = os.path.join(target_folder, framename)


            if not os.path.exists(framename):
                # frame = hs.handsegment(frame)
                lastFrame = frame
                cv2.imwrite(framename, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count += 1
            f.write(framename + os.linesep)
        # repeat last frame untill we get 150 frames
        while count < 150:
            framename = os.path.splitext(video)[0]
            framename = framename + "_frame_" + str(count) + ".png"

            framename = os.path.join(target_folder, framename)

            if not os.path.exists(framename):
                cv2.imwrite(framename, lastFrame)
            count += 1
            f.write(framename + os.linesep)
        cap.release()
        cv2.destroyAllWindows()
    f.close()

    os.chdir(rootPath)



gesture = '/home/quan/PycharmProjects/sign-language-gesture-recognition/data/all'
target =  '/home/quan/PycharmProjects/sign-language-gesture-recognition/data/raw_images'



convert(gesture,target)
# split_train_test_video(root, data_test_video, data_train_video)


