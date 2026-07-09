import argparse
import pandas as pd
import os
import json
import subprocess
from evaluation import metrics_calculator
from tqdm import tqdm

def extract_frames(video_path,store_image_folder):
    if not os.path.exists(store_image_folder):
        os.makedirs(store_image_folder)

    subprocess.run(
        ["ffmpeg", "-i", str(video_path), os.path.join(store_image_folder, "frames_%d.png")],
        check=True,
    )
    

parser = argparse.ArgumentParser()
parser.add_argument("--meta_file", type=str,default="data/evaluation_example/meta_generated.csv")
parser.add_argument("--frame_dir", type=str,default="data/evaluation_example/frames_generated")
parser.add_argument("--gt_meta_file", type=str,default="data/evaluation_example/meta_gt.csv")
parser.add_argument("--gt_frame_dir", type=str,default="data/evaluation_example/frames_gt")
parser.add_argument("--output_folder", type=str,default="data/evaluation_example/results")
parser.add_argument("--ckpt_path", type=str,default="data/ckpt")
parser.add_argument("--device", type=str,default="cuda")
parser.add_argument("--metrics", type=str,nargs='+',default=[
        # temporal consistency
    'temporal_dino_consistency', # ↑
    'temporal_clip_consistency', # ↑
    'temporal_motion_smoothness', # ↑
        # temporal motion strength
    'dynamic_degree', # ↑
    'tracking_strength', # ↑
        # 3D consistency
    '3D_consistency_num_pts', # ↑
    '3D_consistency_num_inliers_F', # ↑
    '3D_consistency_keep_ratio', # ↑
    '3D_consistency_mean_err', # ↓
    '3D_consistency_rmse', # ↓
        # video frame quality
    'aesthetic_quality', # ↑
    'imaging_quality', # ↑
        # text-video alignment
    'camera_alignment', # ↑
    'main_object_alignment', # ↑
    'background_alignment', # ↑
    'style_alignment', # ↑
    'overall_consistency', # ↑
        # distribution consistency
    'fvd&kvd', # ↓
    'fid&kid', # ↓

])

args = parser.parse_args()
meta_file=args.meta_file
frame_dir=args.frame_dir
output_folder=args.output_folder
metrics=args.metrics
device=args.device
gt_meta_file=args.gt_meta_file
ckpt_path=args.ckpt_path
gt_frame_dir=args.gt_frame_dir

meta_info=pd.read_csv(meta_file)

if "fid&kid" in metrics:
    calculate_fid=True
    metrics.remove("fid&kid")
else:
    calculate_fid=False

if "fvd&kvd" in metrics:
    calculate_fvd=True
    metrics.remove("fvd&kvd")
else:
    calculate_fvd=False

My_Metrics_Calculator=metrics_calculator(metrics,ckpt_path=ckpt_path,device=device)

metrics_result_pd=pd.DataFrame(columns=["video_id"]+metrics)

for row_idx in tqdm(range(meta_info.shape[0])):
    present_test_case=meta_info.iloc[row_idx]
    video_idx=present_test_case["video_idx"]
    video_path=present_test_case["video_path"]
    short_caption=present_test_case["short_caption"]
    dense_caption=present_test_case["dense_caption"]
    main_object_caption=present_test_case["main_object_caption"]
    background_caption=present_test_case["background_caption"]
    style_caption=present_test_case["style_caption"]
    camera_caption=present_test_case["camera_caption"]
    
    print(f"================ Video Index {video_idx} ================")
    store_image_folder=os.path.join(frame_dir,str(video_idx))
    if os.path.exists(store_image_folder) and len(os.listdir(store_image_folder))!=0:
        print(f"{store_image_folder} already exists! Please change another folder to avoid overwrite!")
    else:
        print("Extracting frames...")
        extract_frames(video_path,store_image_folder)
        print("Finish extracting frames")

    print(f"================ Calculating Metrics of Index {video_idx} ================")
    present_result=[video_idx]
    
    for metric in metrics:
        try:
            print(f"calculating metrics {metric}")
            present_result.append(My_Metrics_Calculator(metric,store_image_folder,video_path,short_caption,dense_caption,main_object_caption,background_caption,style_caption,camera_caption))
        except Exception as e:
            print(f"Error in calculating metrics {metric}: {e}")
            present_result.append(None)


    metrics_result_pd.loc[len(metrics_result_pd.index)] = present_result
    print(f"Success for video: {video_idx}")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    metrics_result_pd.to_csv(os.path.join(output_folder,"video_score.csv"),index=False)
    print(f'Saved each video score in {os.path.join(output_folder,"video_score.csv")}')

mean_metrics_result_pd=metrics_result_pd.mean()
mean_metrics_result_dict=mean_metrics_result_pd.to_dict()
mean_metrics_result_dict.pop("video_id")

with open(os.path.join(output_folder,"average_score.csv"),"w") as f:
    json.dump(mean_metrics_result_dict,f,indent=4)
print(f'Saved average score in {os.path.join(output_folder,"average_score.csv")}')
print("Finish")

if calculate_fvd or calculate_fid:
    gt_meta_info=pd.read_csv(gt_meta_file)
    for row_idx in range(gt_meta_info.shape[0]):
        present_test_case=gt_meta_info.iloc[row_idx]
        video_idx=present_test_case["video_idx"]
        video_path=present_test_case["video_path"]
        
        store_gt_image_folder=os.path.join(gt_frame_dir,str(video_idx))
        print(f"================ GT Video Index {video_idx} ================")
        if os.path.exists(store_gt_image_folder) and len(os.listdir(store_gt_image_folder))!=0:
            print(f"{store_gt_image_folder} already exists! Please change another folder to avoid overwrite!")
        else:
            print("Extracting frames...")
            extract_frames(video_path,store_gt_image_folder)
            print("Finish extracting frames")

if calculate_fvd:
    try:
        print(f"calculating metrics fvd kvd")
        from evaluation.fvd import EvaluateFVD
        mean_metrics_result_dict["fvd"], mean_metrics_result_dict["kvd"]=EvaluateFVD(frame_dir,gt_frame_dir, ckpt_path, device)
    except Exception as e:
        print(f"Error in calculating metrics fvd kvd: {e}")

with open(os.path.join(output_folder,"average_score.csv"),"w") as f:
    json.dump(mean_metrics_result_dict,f,indent=4)
print(f'Saved average score in {os.path.join(output_folder,"average_score.csv")}')
print("Finish")

if calculate_fid:
    try:
        print(f"calculating metrics fid kid")
        from evaluation.fid import EvaluateFID
        mean_metrics_result_dict["fid"], mean_metrics_result_dict["kid"]=EvaluateFID(frame_dir, gt_frame_dir, ckpt_path, device)
    except Exception as e:
        print(f"Error in calculating metrics fid kid: {e}")

with open(os.path.join(output_folder,"average_score.csv"),"w") as f:
    json.dump(mean_metrics_result_dict,f,indent=4)
print(f'Saved average score in {os.path.join(output_folder,"average_score.csv")}')
print("Finish")
