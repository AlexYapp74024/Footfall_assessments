
#%%
import numpy as np
import pims
from pims import Frame
from slicerator import Slicerator
import cv2
from scipy import ndimage as ndi
from pathlib import Path
import pickle

def cache_opreation(call, path:Path):
    if not path.exists(): 
        obj = call()
        with open(path, "wb") as file: pickle.dump(obj, file)
    
    with open(path, "rb") as file: return pickle.load(file)

@pims.pipeline(ancestor_count = 2)
def frame_diff_xywh(f0:Frame, f1:Frame):
    k = 15
    f0_ = cv2.GaussianBlur(pims.as_gray(f0), (k,k), 9.0)
    f1_ = cv2.GaussianBlur(pims.as_gray(f1), (k,k), 9.0)
    mask_ = np.abs(f0_ - f1_) > 16

    mask_ = ndi.binary_opening(mask_, iterations=1)
    mask_ = ndi.binary_dilation(mask_, ndi.generate_binary_structure(2, 2), iterations=10)

    cnts = cv2.findContours(np.uint8(mask_), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    return np.int32([cv2.boundingRect(c) for c in cnts])

frames = pims.open("sample.mp4")

diff_boxes = frame_diff_xywh(frames[:-1], frames[1:])
frames = frames[1:]

#%%
from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deep_sort.track import Track
from tqdm import tqdm
from itertools import chain

MAX_AGE = 50
from copy import deepcopy
def deepsort(images: list[Frame], boxes: np.ndarray) -> list[Track]:
    tracker = DeepSort(max_age=MAX_AGE)

    def _next(frame:int, image:Frame , boxes: np.ndarray) -> list[Track]:
        detections = [[p, 0.9, "0"] for p in boxes]
        
        tracks : list[Track] = tracker.update_tracks(detections,frame = image, others=boxes)
        for t in tracks:
            if t.det_conf is None: t.det_conf = 0.3
            t.frame = frame
            t.track_id = int(t.track_id)
        return deepcopy(tracks)
    
    tracks = [_next(f, i, r) for f, (i, r) in enumerate(zip(images, tqdm(boxes)))]
    return tracks

def do(): return deepsort(frames, tqdm(diff_boxes))

tracks : list[list[Track]] = cache_opreation(do, Path("output/sample.mp4/deep_track.pyz"))

MIN_AGE = 50
tracks_by_id : dict[str, list[Track]] = {}
for t in chain(*tracks): tracks_by_id[t.track_id] = tracks_by_id.get(t.track_id,[]) + [t]

areas = {id: [np.multiply(*t.to_ltwh()[-2:]) for t in ts] for id, ts in tracks_by_id.items()}
# too_much_size_change = list(id for id, a in areas.items() if max(a)/min(a) > 100)
duration_too_short = list(id for id, ts in tracks_by_id.items() if len(ts) < MAX_AGE + MIN_AGE)
exclude = set(duration_too_short)
tracks = [[t for t in ts if t.track_id not in exclude] for ts in tracks]

#%%
import random
import colorsys
def ramdom_color():
    h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
    r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
    return (r,g,b)
_colors = [ramdom_color() for _ in range(256)]
def colors(i:int): return _colors[int(i) % len(_colors)]

@pims.pipeline(ancestor_count = 3)
def label_tracks(images:Frame,tracks:list[Track], frame = 0, *, thickness=2, font_scale=0.8, orig=True):
    img = images.copy()
    for t in tracks:
        p1,p2 = np.int64(np.reshape(t.to_tlbr(orig=orig),(2,2)))
        img = cv2.rectangle(img,p1,p2,colors(t.track_id),thickness)
        color = colors(int(t.det_class))

        label = f"{t.track_id}:{t.det_conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x,y = p1 - np.array([0,h*2])
        img = cv2.rectangle(img, (x, y - int(h*1.5)), (x + w, y), color, -1)
        img = cv2.putText(img, label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness)
        
    if frame > 0: 
        img = cv2.putText(img, f"{frame}", (10,60),cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
    return Frame(img)

pims.export(label_tracks(frames, Slicerator(tracks)), "out.mp4", 25)