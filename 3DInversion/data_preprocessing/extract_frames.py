import os 
import cv2
import sys

FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
CUR_DIR = os.path.dirname(__file__)
PRJ_DIR = os.path.dirname(CUR_DIR)
RT_DIR = os.path.dirname(PRJ_DIR)
RT_DATA_DIR = os.path.join(RT_DIR, 'data')

sys.path.append(RT_DIR)
sys.path.append(PRJ_DIR)

from modules.utils import load_yaml

config_path = f"{CUR_DIR}/config/{FILE_NAME}.yml"
config = load_yaml(config_path)

DEBUG = config['DEBUG']

PRJ_NAME = config['PATH']['prj_name']
V_SERIAL = config['PATH']['v_serial']
F_SERIAL = config['PATH']['f_serial']

FRAME_NUM = config['SETTING']['frame_num']
DATA_NUM = config['SETTING']['data_num']


if not DEBUG: 
    VIDEO_DATA_DIR = os.path.join(RT_DATA_DIR, PRJ_NAME, f'{V_SERIAL}/downloaded_celebvhq/processed') 
    FRAME_DATA_DIR = os.path.join(RT_DATA_DIR, PRJ_NAME, f'{F_SERIAL}/frame_imgs')
else:
    VIDEO_DATA_DIR = os.path.join(RT_DATA_DIR, PRJ_NAME, f'debug/downloaded_celebvhq/processed') 
    FRAME_DATA_DIR = os.path.join(RT_DATA_DIR, PRJ_NAME, f'debug/frame_imgs')

os.makedirs(FRAME_DATA_DIR, exist_ok=True)
v_fname_list = os.listdir(VIDEO_DATA_DIR)
print("start extracting")
for idx, v_fname in enumerate(v_fname_list): 
    if idx > DATA_NUM: break
    v_id = "_".join(s for s in v_fname.split("_")[:-1])
    v_path = f"{VIDEO_DATA_DIR}/{v_fname}"
    print(idx, ":", v_id)
    
    frame_dir = f"{FRAME_DATA_DIR}/{v_id}"
    os.makedirs(frame_dir, exist_ok=True)
   
    video = cv2.VideoCapture(v_path)
    if not video.isOpened():
        print("Could not Open :", v_path)
        # exit(0)
        continue
        
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(video.get(cv2.CAP_PROP_FPS))
    # print("fps done")

#     print("length :", length)
#     print("width :", width)
#     print("height :", height)
#     print("fps :", fps)
    
    count = 0
    tmp_length = 0

    period = length // FRAME_NUM

    while(video.isOpened()):
        ret, image = video.read()

        if FRAME_NUM:
            if (int(video.get(1)) % (tmp_length+period) == 0):
                frame_path = f"{frame_dir}/frame{count}.jpg"
                cv2.imwrite(frame_path, image)
                tmp_length += period
                count += 1

                if abs(tmp_length - length) < period:
                    break
                if count == FRAME_NUM:
                    break
        else:
            if(int(video.get(1)) % fps == 0): #앞서 불러온 fps 값을 사용하여 1초마다 추출
                # print(f"frame no{count}")
                frame_path = f"{frame_dir}/frame{count}.jpg"
                cv2.imwrite(frame_path, image)
    #             print('Saved frame number :', str(int(video.get(1))))
                count += 1
                tmp_length += fps

                if abs(tmp_length - length) < fps:
                    break
    # print("done")
    video.release()