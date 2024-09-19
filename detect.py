#%% Imports
from pathlib import Path
import random
from slicerator import Slicerator
from tqdm import tqdm
import cv2
import numpy as np
from pims import Frame
import pims
from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
import sahi.models
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction
from ultralytics import YOLO, FastSAM
from ultralytics.engine.results import Boxes, Results, Masks
import pickle
import random
import colorsys
#%%
frames = pims.open("sample.mp4")

def ramdom_color():
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return (r,g,b)

_colors = [ramdom_color() for _ in range(256)]
def colors(i:int): return _colors[int(i) % len(_colors)]

def animate_images(images: list[np.ndarray], cmap = None):
    images = list(images)
    fig = plt.figure()
    plt.axis('off')
    im = plt.imshow(images[0], cmap)
    def animate(k):
        im.set_array(images[k])
        return im,
    ani = animation.FuncAnimation(fig, animate, frames=len(images), blit=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
    plt.close()
    return HTML(ani.to_jshtml())

@pims.pipeline(ancestor_count = 3)
def label_sahi_results(image:Frame, preds:list[ObjectPrediction], frame = 0, thickness=2, font_scale=0.8, lable_frame = True):
    img = image
    for p in preds:
        id = p.category.id
        p1,p2 = np.int64(np.reshape(p.bbox.to_xyxy(),(2,2)))
        color = colors(id)
        img = cv2.rectangle(img,p1,p2,color,thickness)
        
        label = f"{p.category.name}:{p.score.value:.2f}"

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x,y = p1 - np.array([0,h*2])
        img = cv2.rectangle(img, (x, y - h), (x + w, y), color, -1)
        img = cv2.putText(img, label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
                
    if lable_frame: 
        img = cv2.putText(img, f"{frame}", (10,60),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
    return Frame(img)

def cache_opreation(call, path:Path):
    if not path.exists(): 
        obj = call()
        with open(path, "wb") as file: pickle.dump(obj, file)
    
    with open(path, "rb") as file: return pickle.load(file)

@pims.pipeline
def predict_sliced(frame) -> list[ObjectPrediction]:
    return get_sliced_prediction(frame, sahi_model, 240, 240, verbose=False).object_prediction_list

video_path = Path("sample.mp4")
frames = pims.open(video_path.__str__())[:-1]
(RESULT_DIR := Path("output") / video_path.name).mkdir(exist_ok=True,parents=True)

def do(): return [r for r in tqdm(predict_sliced(frames))]

sahi_model = sahi.AutoDetectionModel.from_pretrained("yolov8", "detect_badge.pt")
badge_boxes : list[list[ObjectPrediction]] = Slicerator(cache_opreation(do, RESULT_DIR / "badges.pyz"))

indices = Slicerator(range(len(frames)))
labeled = label_sahi_results(frames, badge_boxes, indices)
pims.export(labeled, "badges.mp4", 25)
#%%
frames_with_badges = [i for i, b in enumerate(badge_boxes) if len(b)]
ranges = []
start = prev = frames_with_badges[0]
for i in frames_with_badges[1:]:
    if i != prev + 1:
        ranges.append((start,prev))
        start = i
    prev = i
print(", ".join([f"{start}-{end}" for start,end in ranges]))
