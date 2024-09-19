#%%
from pathlib import Path
from joblib import delayed, Parallel
from tqdm import tqdm
import shutil
import random
import cv2
from patchify import patchify
import numpy as np

#%% 
random.seed(42)
(RAW_DATA_DIR := Path("dataRaw/badge")).mkdir(exist_ok=True)
DATASET_DIR = Path("dataset/badge")
shutil.rmtree(DATASET_DIR,True)
DATASET_DIR.mkdir()

labels = list(Path(RAW_DATA_DIR / "labels").glob("*"))
labels = random.sample(labels, len(labels))

size  = len(labels)
train = labels[:int(0.85 * size)]
valid = labels[len(train):]

assert train + valid == labels
#%%
def cxywh_from_txt(string:str):
    inputs = string.split(" ")
    assert len(inputs) == 5
    return np.array([float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3]), float(inputs[4])])

def patchify_xywh(xywhs: list[np.ndarray], img_patch_shape: tuple[int], step=1):
    rows, cols = img_patch_shape[:2]
    ih, iw = img_patch_shape[3:5]
    
    x_steps = [range(x*step, x*step + iw) for x in range(cols)]
    y_steps = [range(y*step, y*step + ih) for y in range(rows)]
    xywh_patches = [[[] for _ in range(cols)] for _ in range(rows)]

    for xywh in xywhs:
        c,x,y,w,h = xywh
        rows_ = [(i,gap.start) for i, gap in enumerate(x_steps) if x in gap and x+w in gap]
        cols_ = [(i,gap.start) for i, gap in enumerate(y_steps) if y in gap and y+h in gap]
        for r_, x_start in rows_: 
            for c_, y_start in cols_:
                xywh_patches[c_][r_].append(np.array([c,x-x_start,y-y_start,w,h]))

    return xywh_patches

def create_dataset(labels:list[Path], dir:str):
    out_dir = DATASET_DIR / dir
    (out_label_dir := out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_image_dir := out_dir / "images").mkdir(parents=True, exist_ok=True)

    PW, PH = 240, 240
    step=120

    def _create_dataset(label_path: Path):
        image_path = label_path.parents[1] / "images" / (label_path.stem + ".webp")
        image = cv2.imread(str(image_path))
        
        ih,iw,_ = image.shape
        xywh_const = [1,iw,ih,iw,ih]
        xywhs = [np.int32(cxywh_from_txt(row) * xywh_const) for row in label_path.read_text().split("\n") if len(row)]

        img_patches = patchify(image, (PW,PH,3), step=step)
        cxywh_patch  = patchify_xywh(xywhs, img_patches.shape, step=step)
        
        img_patches = np.reshape(img_patches, (-1,) + img_patches.shape[3:])
        digits = len(str(img_patches.shape[0]))
        for i, (img, cxywhs) in enumerate(zip(img_patches, sum(cxywh_patch, []))):
            filename = f"{label_path.stem}-{str(i).zfill(digits)}"
            
            with open(out_label_dir / f"{filename}.txt", "w") as file: 
                file.writelines([f"{c} {x/PW} {y/PH} {w/PW} {h/PH}" for c,x,y,w,h in cxywhs])

            cv2.imwrite(str(out_image_dir / f"{filename}.webp"), img)

    Parallel(-1)(delayed(_create_dataset)(l) for l in tqdm(labels))
    
create_dataset(train, "train")
create_dataset(valid, "valid")

#%%
import yaml
with open(RAW_DATA_DIR / "data.yaml") as file:
    yaml_data = yaml.safe_load(file)

yaml_data["train"]= "train/images"
yaml_data["val"] =  "valid/images"

with open(DATASET_DIR / "data.yaml", "w") as file:
    yaml.safe_dump(yaml_data, file)

# %%
from pathlib import Path
DATASET_DIR = Path("dataset")

params = f"batch=32 epochs=70 imgsz=256 dropout=0.3"
print(f"yolo train model=yolov8n.pt data={(DATASET_DIR/'data.yaml').absolute()} {params}")