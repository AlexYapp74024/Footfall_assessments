#%% 
import pims
from pims import Frame
from slicerator import Slicerator
import cv2
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed

#%%
frames = pims.open("sample.mp4")
DATA_DIR = Path("dataraw")

digits = len(str(len(frames)))
@pims.pipeline(ancestor_count = 2)
def copy_images(image:Frame, frame:int, dir:str):
    filename = f"Image---{str(frame).zfill(digits)}"
    out = (DATA_DIR / dir / filename).with_suffix(".webp")
    if not out.exists(): 
        out.parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(str(out),cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

#%%
indices = Slicerator(range(len(frames)))
def do(d, pb:tqdm): d, pb.update(1)

dataset = copy_images(frames, indices, "badge/images")[::3]
with tqdm(total = len(dataset)) as pb:
    Parallel(-1, require="sharedmem")(delayed(do)(d, pb) for d in dataset)