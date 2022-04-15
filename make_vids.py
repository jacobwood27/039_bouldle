import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

from platform import release
import cv2 as cv
import glob
import numpy as np
from skimage import morphology
from skimage.morphology import disk, opening, closing
import json
from skimage.measure import label
import torch
import multiprocessing as mp

IDS = [
    106036163,
    107884306,
    106244180,
    106551257,
    106390524,
    108312154]



def get_climb(id):
    with open('stats.js') as dataFile:
        data = dataFile.read()
        obj = data[data.find('{') : data.rfind('}')+1]
        jsonObj = json.loads(obj)
        return jsonObj[str(id)]

def download_vid(url, dir):
    cmd = "youtube-dl -o " + dir + "/orig " + url
    os.system(cmd)

def get_median_frame(frames):    
    return np.median(frames, axis=0).astype(dtype=np.uint8)

def get_frames(cap):
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def get_median_from_cap(cap):
    #just go 50 frames at a time and median of medians
    bigL = []
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            if len(frames) > 0:
                bigL.append(np.median(frames, axis=0).astype(dtype=np.uint8))
            break
        frames.append(frame)
        if len(frames) == 50:
            bigL.append(np.median(frames, axis=0).astype(dtype=np.uint8))
            frames = []
    return np.median(bigL, axis=0).astype(dtype=np.uint8)


def get_fg_mask(frame, backSub):

    temp = frame.copy()
    fg_mask = backSub.apply(frame,temp,0)
    og_mask = fg_mask.copy()

    fg_mask[fg_mask > 1] = 255

    inter = cv.morphologyEx(fg_mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=2)
    inter = cv.morphologyEx(inter,  cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=2)
    cnts, _ = cv.findContours(inter, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv.contourArea)

    out = np.zeros(fg_mask.shape, np.uint8)
    cv.drawContours(out, cnts, -1, 255, cv.FILLED)
    out = cv.bitwise_and(og_mask, out)


    # blur = cv.GaussianBlur(fg_mask, (3,3), 0)
    # thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for c in cnts:
    #     area = cv.contourArea(c)
    #     if area < 5500:
    #         cv.drawContours(thresh, [c], -1, (0,0,0), -1)

    # kernel = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    # close = 255 - cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations=2)

    # footprint = disk(4)
    # fg_mask = opening(fg_mask, footprint)

    # processed = morphology.remove_small_objects(fg_mask.astype(np.bool), min_size=100).astype(int)
    # mask_x, mask_y = np.where(processed == 0)
    # fg_mask[mask_x, mask_y] = 0
    # return fg_mask
    return out

def get_hint_im(median_frame, fg_mask):
    hint_im = cv.bitwise_and(median_frame, median_frame, mask=~fg_mask)
    return hint_im

def process_frame(frame, median_frame, work_dir, num):

    backSub = cv.createBackgroundSubtractorKNN(detectShadows=True)
    [backSub.apply(median_frame) for i in range(10)]

    fg_mask = get_fg_mask(frame, backSub)
    hint_im = get_hint_im(median_frame, fg_mask)

    cv.imwrite(work_dir + "/hidden/" + f'{num:05}' +".png", 255 - fg_mask)
    cv.imwrite(work_dir + "/hint/" + f'{num:05}' +".png", hint_im)

def make_vids(work_dir,out_dir):

    if os.path.isfile(work_dir + "/orig.mkv"):
        vid_file = work_dir + "/orig.mkv"
    else:
        vid_file = work_dir + "/orig.mp4"

    if not os.path.isfile(work_dir + "/medianframe.png"):
        print("Determining median frame")
        cap = cv.VideoCapture(vid_file)
        median_frame = get_median_from_cap(cap)
        cap.release()
        cv.imwrite(work_dir + "/medianframe.png", median_frame)        

    cap = cv.VideoCapture(vid_file)
    vh = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    vw = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    cap.release()

    if not os.path.isdir(work_dir + "/rvm_matting/com"):

        print("Running RVM Matting")
        # model = torch.hub.load("PeterL1n/RobustVideoMatting", "resnet50") # or "mobilenetv3"
        # convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")
        # convert_video(
        #     model,                           # The loaded model, can be on any device (cpu or cuda).
        #     device='cuda',
        #     input_source=vid_file,           # A video file or an image sequence directory.
        #     downsample_ratio=None,       # [Optional] If None, make downsampled max size be 512px.
        #     output_type='video',             # Choose "video" or "png_sequence"
        #     output_composition= work_dir + "/rvm_matting/com.mp4",    # File path if video; directory path if png sequence.
        #     output_alpha= work_dir + "/rvm_matting/pha.mp4",          # [Optional] Output the raw alpha prediction.
        #     output_foreground= work_dir + "/rvm_matting/fgr.mp4",     # [Optional] Output the raw foreground prediction.
        #     output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
        #     seq_chunk=1,                    # Process n frames at once for better parallelism.
        #     num_workers=1,                   # Only for image sequence input. Reader threads.
        #     progress=True                    # Print conversion progress.
        # )
        cmd = """
                python working/RobustVideoMatting/inference.py \\
                --variant resnet50 \\
                --checkpoint "/home/jwood/.cache/torch/hub/checkpoints/rvm_resnet50.pth" \\
                --device cpu \\
                --input-source "{src}" \\
                --output-type png_sequence \\
                --output-composition "{out_com}" \\
                --output-alpha "{out_alph}" \\
                --output-foreground "{out_fgr}" \\
                --seq-chunk 24
            """.format(
            src=vid_file, 
            ds_rat="None",
            out_com=work_dir + "/rvm_matting/com/",
            out_alph= work_dir + "/rvm_matting/alpha/",
            out_fgr=work_dir + "/rvm_matting/fgr/")
        os.system(cmd)

    if not os.path.isfile(work_dir + "/bgmv2_matting/com.mp4"):
        print("Running BGMV2 Matting")

        if vw > 3500:
            scale = 0.25
            pix = 600000
        elif vw > 2000:
            scale = 0.5
            pix = 400000
        else:
            scale = 0.5
            pix = 200000

        cmd = """
        python working/BackgroundMattingV2/inference_video.py \\
        --model-type mattingrefine \\
        --model-backbone resnet50 \\
        --model-backbone-scale {scale} \\
        --model-refine-mode sampling \\
        --model-refine-sample-pixels {pix} \\
        --model-checkpoint "working/BackgroundMattingV2/pytorch_resnet50.pth" \\
        --video-src "{src}" \\
        --video-bgr "{bgr}" \\
        --output-dir "{output_dir}" \\
        --output-type com fgr pha err ref
        """.format(
            scale=str(scale), 
            pix=str(pix),
            src=vid_file,
            output_dir= work_dir + "/bgmv2_matting/",
            bgr=work_dir + "/medianframe.png")

        os.system(cmd)






    # print("Processing Frames")
    # pool = mp.Pool(processes=12)
    # pool.starmap(process_frame, [(f, median_frame, work_dir, i) for i,f in enumerate(frames)])

    # print("Compiling into final videos")
    # cmds = [
    #     "ffmpeg -an -y -i " + work_dir + "/hidden/%05d.png -vcodec libx264    -crf 30 -pix_fmt yuv420p -profile:v baseline -level 3 -movflags +faststart " + out_dir + "/hidden.mp4",
    #     "ffmpeg -an -y -i " + work_dir + "/hidden/%05d.png -vcodec libvpx-vp9 -crf 63 -b:v 0 -movflags +faststart " + out_dir + "/hidden.webm",
    #     "ffmpeg -an -y -i " + work_dir + "/hint/%05d.png   -vcodec libx264    -crf 30 -pix_fmt yuv420p -profile:v baseline -level 3 -movflags +faststart " + out_dir + "/hint.mp4",
    #     "ffmpeg -an -y -i " + work_dir + "/hint/%05d.png   -vcodec libvpx-vp9 -crf 63 -b:v 0 -movflags +faststart " + out_dir + "/hint.webm",
    # ]
    # for cmd in cmds:
    #     os.system(cmd)
        

def track_vid(work_dir):
    cap = cv.VideoCapture(work_dir + "/orig.mp4")
    tracker = cv.TrackerCSRT_create()
    initBB = None
    count = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        count += 1
        if initBB is not None:
            (success, box) = tracker.update(frame)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv.rectangle(frame, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)

        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("s"):
            initBB = cv.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
            tracker.init(frame, initBB)

    cv.destroyAllWindows()

for ID in IDS:
    climb = get_climb(ID)
    print(ID, climb["Name"], climb["src"])
    
    work_dir   = "working/vids/" + str(ID)
    result_dir = "vids/" + str(ID)

    if not os.path.isdir(work_dir):
        os.mkdir(work_dir)
    
    if not os.path.isdir(work_dir + "/hint"):
        os.mkdir(work_dir + "/hint")
    
    if not os.path.isdir(work_dir + "/hidden"):
        os.mkdir(work_dir + "/hidden")

    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    if not (os.path.isfile(work_dir + "/orig.mp4") or os.path.isfile(work_dir + "/orig.mkv")):
        download_vid(climb["src"], work_dir)
    
    # track_vid(work_dir)
    make_vids(work_dir, result_dir)
