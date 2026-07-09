import gc
import random
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from datetime import datetime
import threading

from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from diffusers import AutoencoderKL, DDIMScheduler, EulerDiscreteScheduler
from util.stable_diffusion_inpaint import StableDiffusionInpaintPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from marigold_lcm.marigold_pipeline import MarigoldPipeline, MarigoldPipelineNormal, MarigoldNormalsPipeline

from models.models import KeyframeGen, save_point_cloud_as_ply
from util.gs_utils import save_pc_as_3dgs, convert_pc_to_splat
from util.chatGPT4 import TextpromptGen
from util.general_utils import apply_depth_colormap, save_video
from util.utils import save_depth_map, prepare_scheduler, soft_stitching
from util.utils import load_inference_prompt_config, convert_pt3d_cam_to_3dgs_cam, sky_image_dir
from util.segment_utils import create_mask_generator_repvit
from util.free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
 
from arguments import GSParams, CameraParams
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.loss import l1_loss, ssim
from scene.cameras import Camera
from random import randint
import time
from syncdiffusion.syncdiffusion_model import SyncDiffusion
from kornia.morphology import dilation
import warnings
import os
import sys
import copy
warnings.filterwarnings("ignore")


class BatchEmitter:
    def emit(self, *_args, **_kwargs):
        return None


def _ckpt_root() -> Path:
    return Path(os.environ.get("WORLDFOUNDRY_CKPT_DIR", Path(__file__).resolve().parents[6] / "ckpt"))


def _local_model_ref(default: str, *candidates: str) -> str:
    for candidate in candidates:
        path = _ckpt_root() / candidate
        if path.exists():
            return str(path)
    return default


socketio = BatchEmitter()

xyz_scale = 1000
client_id = None
scene_name = None
view_matrix = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_wonder = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
view_matrix_delete = [-1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

view_matrix_fixed = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0.2, 0.5, 1]
])
theta = np.radians(-3)
rotation_matrix_x = np.array([
    [1, 0, 0, 0],
    [0, np.cos(theta), -np.sin(theta), 0],
    [0, np.sin(theta), np.cos(theta), 0],
    [0, 0, 0, 1]
])
view_matrix_fixed = np.dot(view_matrix_fixed, rotation_matrix_x)
view_matrix_fixed = view_matrix_fixed.flatten().tolist()

background = torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32, device='cuda')
latest_frame = None
latest_viz = None
keep_rendering = True
iter_number = None
kf_gen = None
gaussians = None
opt = None
scene_dict = None
style_prompt = None
pt_gen = None
change_scene_name_by_user = False
undo = False
save = False
delete = False
exclude_sky = False

def empty_cache():
    torch.cuda.empty_cache()
    gc.collect()


def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")


def _worldfoundry_batch_mode(config):
    if "worldfoundry_interactive" in config:
        return not bool(config["worldfoundry_interactive"])
    if "worldfoundry_batch" in config:
        return bool(config["worldfoundry_batch"])
    return True


def _worldfoundry_iterations(config, key, default):
    try:
        return int(config.get(key, config.get("worldfoundry_gs_iterations", default)))
    except Exception:
        return int(default)


def _save_worldfoundry_batch_output(config, gaussians):
    save_dir = Path(config["runs_dir"]) / "worldfoundry_batch"
    save_dir.mkdir(parents=True, exist_ok=True)
    gaussians.save_ply_all_with_filter(save_dir / "finished_3dgs.ply")
    torch.save(gaussians.visibility_filter_all, save_dir / "visibility_filter_all.pth")
    torch.save(gaussians.is_sky_filter, save_dir / "is_sky_filter.pth")
    torch.save(gaussians.delete_mask_all, save_dir / "delete_mask_all.pth")
    try:
        gaussians.yield_splat_data(save_dir / "finished_3dgs.splat")
    except Exception as exc:
        print(f"WorldFoundry batch splat export skipped: {exc}")
    print(f"WorldFoundry batch output saved to {save_dir}")


def run(config):
    global client_id, view_matrix, scene_name, latest_frame, keep_rendering, kf_gen, latest_viz, gaussians, opt, background, scene_dict, style_prompt, pt_gen, change_scene_name_by_user, undo, save, delete, exclude_sky, view_matrix_delete

    ###### ------------------ Load modules ------------------ ######

    seeding(config["seed"])
    example = config['example_name']
    sky_dir = sky_image_dir(example)

    oneformer_ref = _local_model_ref("shi-labs/oneformer_ade20k_swin_large", "oneformer_ade20k_swin_large")
    segment_processor = OneFormerProcessor.from_pretrained(oneformer_ref)
    segment_model = OneFormerForUniversalSegmentation.from_pretrained(oneformer_ref).to('cuda')

    mask_generator = create_mask_generator_repvit()

    inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config["stable_diffusion_checkpoint"],
            safety_checker=None,
            torch_dtype=torch.bfloat16,
        ).to(config["device"])
    inpainter_pipeline.scheduler = DDIMScheduler.from_config(inpainter_pipeline.scheduler.config)
    inpainter_pipeline.unet.set_attn_processor(AttnProcessor2_0())
    inpainter_pipeline.vae.set_attn_processor(AttnProcessor2_0())
    
    rotation_path = config['rotation_path'][:config['num_scenes']]
    assert len(rotation_path) == config['num_scenes']
    
    
    depth_model = MarigoldPipeline.from_pretrained(
        _local_model_ref("prs-eth/marigold-depth-v1-0", "marigold-depth-v1-0"),
        torch_dtype=torch.bfloat16,
    ).to(config["device"])
    depth_model.scheduler = EulerDiscreteScheduler.from_config(depth_model.scheduler.config)
    depth_model.scheduler = prepare_scheduler(depth_model.scheduler)

    normal_estimator = MarigoldNormalsPipeline.from_pretrained(
        _local_model_ref("prs-eth/marigold-normals-v0-1", "prs-eth--marigold-normals-v0-1"),
        torch_dtype=torch.bfloat16,
    ).to(config["device"])
    
    print('###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######') 
    kf_gen = KeyframeGen(config=config, inpainter_pipeline=inpainter_pipeline, mask_generator=mask_generator, depth_model=depth_model,
                            segment_model=segment_model, segment_processor=segment_processor, normal_estimator=normal_estimator,
                            rotation_path=rotation_path, inpainting_resolution=config['inpainting_resolution_gen']).to(config["device"])

    yaml_data = load_inference_prompt_config(config)
    content_prompt, style_prompt, adaptive_negative_prompt, background_prompt, control_text, outdoor = yaml_data['content_prompt'], yaml_data['style_prompt'], yaml_data['negative_prompt'], yaml_data.get('background', None), yaml_data.get('control_text', None), yaml_data.get('outdoor', False)
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "

    start_keyframe = Image.open(yaml_data['image_filepath']).convert('RGB').resize((512, 512))
    kf_gen.image_latest = ToTensor()(start_keyframe).unsqueeze(0).to(config['device'])
    
    if config['gen_sky_image'] or (not (sky_dir / "sky_0.png").exists() and not (sky_dir / "sky_1.png").exists()):
        syncdiffusion_model = SyncDiffusion(
            config['device'],
            sd_version='2.0-inpaint',
            hf_key=config.get("stable_diffusion_checkpoint"),
        )
    else:
        syncdiffusion_model = None
    sky_mask = kf_gen.generate_sky_mask().float()
    kf_gen.generate_sky_pointcloud(syncdiffusion_model, image=kf_gen.image_latest, mask=sky_mask, gen_sky=config['gen_sky_image'], style=style_prompt)

    kf_gen.recompose_image_latest_and_set_current_pc(scene_name=scene_name)
    
    pt_gen = TextpromptGen(kf_gen.run_dir, isinstance(control_text, list))
    
    content_list = content_prompt.split(',')
    scene_name = content_list[0]
    entities = content_list[1:]
    scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}
    inpainting_prompt = content_prompt
    socketio.emit('scene-prompt', scene_name, room=client_id)

    kf_gen.increment_kf_idx()
    ###### ------------------ Main loop ------------------ ######

    if config['gen_sky'] or not (sky_dir / "finished_3dgs_sky_tanh.ply").exists():
        traindatas = kf_gen.convert_to_3dgs_traindata(xyz_scale=xyz_scale, remove_threshold=None, use_no_loss_mask=False)
        if config['gen_layer']:
            traindata, traindata_sky, traindata_layer = traindatas
        else:
            traindata, traindata_sky = traindatas
        gaussians = GaussianModel(sh_degree=0, floater_dist2_threshold=9e9)
        opt = GSParams()
        opt.max_screen_size = 100  # Sky is supposed to be big; set a high max screen size
        opt.scene_extent = 1.5  # Sky is supposed to be big; set a high scene extent
        opt.densify_from_iter = 200  # Need to do some densify
        opt.prune_from_iter = 200  # Don't prune for sky because sky 3DGS are supposed to be big; prevent it by setting a high prune iter
        opt.densify_grad_threshold = 1.0  # Do not need to densify; Set a high threshold to prevent densifying
        opt.iterations = _worldfoundry_iterations(config, "worldfoundry_sky_iterations", 399)  # More iterations than 100 needed for sky
        scene = Scene(traindata_sky, gaussians, opt, is_sky=True)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_sky"
        optimize_gaussians(gaussians, scene, opt, save_dir, initialize_scaling=False)
        sky_dir.mkdir(parents=True, exist_ok=True)
        gaussians.save_ply_with_filter(sky_dir / "finished_3dgs_sky_tanh.ply")
    else:
        gaussians = GaussianModel(sh_degree=0)
        gaussians.load_ply_with_filter(sky_dir / "finished_3dgs_sky_tanh.ply")  # pure sky

    gaussians.visibility_filter_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    gaussians.delete_mask_all = torch.zeros(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    gaussians.is_sky_filter = torch.ones(gaussians.get_xyz_all.shape[0], dtype=torch.bool, device='cuda')
    
    if config['load_gen'] and (sky_dir / "finished_3dgs.ply").exists() and (sky_dir / "visibility_filter_all.pth").exists() and (sky_dir / "is_sky_filter.pth").exists() and (sky_dir / "delete_mask_all.pth").exists():
        print("Loading existing 3DGS...")
        gaussians = GaussianModel(sh_degree=0)
        gaussians.load_ply_with_filter(sky_dir / "finished_3dgs.ply")
        gaussians.visibility_filter_all = torch.load(sky_dir / "visibility_filter_all.pth").to('cuda')
        gaussians.is_sky_filter = torch.load(sky_dir / "is_sky_filter.pth").to('cuda')
        gaussians.delete_mask_all = torch.load(sky_dir / "delete_mask_all.pth").to('cuda')
    opt = GSParams()
    if _worldfoundry_batch_mode(config):
        opt.iterations = _worldfoundry_iterations(config, "worldfoundry_gs_iterations", opt.iterations)

    ### First scene 3DGS
    if config['gen_layer']:
        traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata_layer, gaussians, opt)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_layer{0:02d}"
        optimize_gaussians(gaussians, scene, opt, save_dir)  # Base layer optimization
    else:
        traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)

    gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
    scene = Scene(traindata, gaussians, opt)
    dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    i = 0
    save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{i:02d}"
    optimize_gaussians(gaussians, scene, opt, save_dir)

    tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_at_origin(), xyz_scale=xyz_scale)
    gaussians.set_inscreen_points_to_visible(tdgs_cam)

    if _worldfoundry_batch_mode(config):
        _save_worldfoundry_batch_output(config, gaussians)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    
    def llm_prompt_generation(event):
        global scene_dict, style_prompt, pt_gen, change_scene_name_by_user, scene_name
        while True:
            event.wait()
            print("-- start llm...")
            scene_dict = pt_gen.wonder_next_scene(scene_name=scene_name, entities=scene_dict['entities'], style=style_prompt, background=scene_dict['background'], change_scene_name_by_user=change_scene_name_by_user)
            change_scene_name_by_user = False
            print("-- llm done.")
            event.clear()
        
    if config['use_gpt']:
        llm_event = threading.Event()
        llm_thread = threading.Thread(target=llm_prompt_generation, args=(llm_event, ))
        llm_thread.daemon = True
        llm_thread.start()
    
    gaussians_tmp = copy.deepcopy(gaussians)
    while True:
        inpainting_prompt = pt_gen.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])
        scene_name = scene_dict['scene_name'] if isinstance(scene_dict['scene_name'], str) else scene_dict['scene_name'][0]
        i += 1
        
        socketio.emit('scene-prompt', scene_name, room=client_id)
        print('Waiting for scene gen signal...')
        socketio.emit('server-state', 'Waiting to generate new scenes...', room=client_id)
        
        while keep_rendering:
            time.sleep(0.05)
            if delete:
                print("Deleting...")
                current_pt3d_cam_delete = kf_gen.get_camera_by_js_view_matrix(view_matrix_delete, xyz_scale=xyz_scale)
                tdgs_cam_delete = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam_delete, xyz_scale=xyz_scale)
                gaussians.delete_points(tdgs_cam_delete)
                delete = False
            if save:
                print("Saving...")
                sky_dir.mkdir(parents=True, exist_ok=True)
                gaussians.save_ply_all_with_filter(sky_dir / "finished_3dgs.ply")
                torch.save(gaussians.visibility_filter_all, sky_dir / "visibility_filter_all.pth")
                torch.save(gaussians.is_sky_filter, sky_dir / "is_sky_filter.pth")
                torch.save(gaussians.delete_mask_all, sky_dir / "delete_mask_all.pth")
                gaussians.yield_splat_data(sky_dir / f"{example}_finished_3dgs.splat")
                save = False
        
        if undo:
            print("Undoing...")
            gaussians = copy.deepcopy(gaussians_tmp)
            undo = False
        else:
            print("Not undo...")
            gaussians_tmp = copy.deepcopy(gaussians)
             
        socketio.emit('server-state', 'Generating new scene...', room=client_id)
        
        # LLM prompt generation
        if config['use_gpt']:
            llm_event.set()
        
        if config['use_gpt']:
            scene_dict = pt_gen.wonder_next_scene(scene_name=scene_name, entities=scene_dict['entities'], style=style_prompt, background=scene_dict['background'], change_scene_name_by_user=change_scene_name_by_user)
            change_scene_name_by_user = False
        inpainting_prompt = pt_gen.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])
        scene_name = scene_dict['scene_name'] if isinstance(scene_dict['scene_name'], str) else scene_dict['scene_name'][0]
        
        ###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######        
        kf_gen.set_kf_param(inpainting_resolution=config['inpainting_resolution_gen'],
                            inpainting_prompt=inpainting_prompt, adaptive_negative_prompt=adaptive_negative_prompt)
        current_pt3d_cam = kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale)
        tdgs_cam = convert_pt3d_cam_to_3dgs_cam(current_pt3d_cam, xyz_scale=xyz_scale)
        kf_gen.set_current_camera(current_pt3d_cam, archive_camera=True)
        
        if exclude_sky:
            with torch.no_grad():
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)
            
            side_sky_height = 128

            inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"]<0.6)
            inpaint_mask_0p0_nosky = (render_pkg_nosky["final_opacity"]<0.01)  # Should not have holes in existing regions
            inpaint_mask_0p5 = (render_pkg["final_opacity"]<0.6)
            inpaint_mask_0p0 = (render_pkg["final_opacity"]<0.01)  # Should not have holes in existing regions

            mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
            mask_using_full_render[:, :, :side_sky_height, :] = 1
            
            mask_using_nosky_render = 1 - mask_using_full_render
                
            outpaint_condition_image = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
            viz = outpaint_condition_image[0].permute(1, 2, 0).detach().cpu().numpy()
            viz = (viz * 255).astype(np.uint8)
            viz = viz[..., ::-1]
            # latest_viz = viz
            fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
            outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())
            exclude_sky = False
        else:
            with torch.no_grad():
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                render_pkg_nosky = render(tdgs_cam, gaussians, opt, background, exclude_sky=True)
            
            side_sky_height = 128
            sky_cond_width = 40

            inpaint_mask_0p5_nosky = (render_pkg_nosky["final_opacity"]<0.6)
            inpaint_mask_0p0_nosky = (render_pkg_nosky["final_opacity"]<0.01)  # Should not have holes in existing regions
            inpaint_mask_0p5 = (render_pkg["final_opacity"]<0.6)
            inpaint_mask_0p0 = (render_pkg["final_opacity"]<0.01)  # Should not have holes in existing regions
            fg_mask_0p5_nosky = ~inpaint_mask_0p5_nosky.clone()
            foreground_cols = torch.sum(fg_mask_0p5_nosky == 1, dim=1)>150  # [1, 512]
            foreground_cols_idx = torch.nonzero(foreground_cols, as_tuple=True)[1]

            mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
            if foreground_cols_idx.numel() > 0:
                min_index = foreground_cols_idx.min().item()
                max_index = foreground_cols_idx.max().item()
                mask_using_full_render[:, :, :, min_index:max_index+1] = 1
            mask_using_full_render[:, :, :sky_cond_width, :] = 1
            mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
            mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1
            
            mask_using_nosky_render = 1 - mask_using_full_render
                
            outpaint_condition_image = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
            viz = outpaint_condition_image[0].permute(1, 2, 0).detach().cpu().numpy()
            viz = (viz * 255).astype(np.uint8)
            viz = viz[..., ::-1]
            # latest_viz = viz
            fill_mask = inpaint_mask_0p5_nosky * mask_using_nosky_render + inpaint_mask_0p5 * mask_using_full_render
            outpaint_mask = inpaint_mask_0p0_nosky * mask_using_nosky_render + inpaint_mask_0p0 * mask_using_full_render
            outpaint_mask = dilation(outpaint_mask, kernel=torch.ones(7, 7).cuda())

        
        inpaint_output = kf_gen.inpaint(outpaint_condition_image, inpaint_mask=outpaint_mask, fill_mask=fill_mask, inpainting_prompt=inpainting_prompt, mask_strategy=np.max, diffusion_steps=50)

        sem_seg = kf_gen.update_sky_mask()
        recomposed = soft_stitching(render_pkg["render"], kf_gen.image_latest, kf_gen.sky_mask_latest)  # Replace generated sky with rendered sky

        depth_should_be = render_pkg['median_depth'][0:1].unsqueeze(0) / xyz_scale
        mask_to_align_depth = (depth_should_be < 0.006 * 0.8) & (depth_should_be > 0.001)  # If opacity < 0.5, then median_depth = -1

        ground_mask = kf_gen.generate_ground_mask(sem_map=sem_seg)[None, None]
        depth_should_be_ground = kf_gen.compute_ground_depth(camera_height=0.0003)
        ground_outputable_mask = (depth_should_be_ground > 0.001) & (depth_should_be_ground < 0.006 * 0.8)

        joint_mask = mask_to_align_depth | (ground_mask & ground_outputable_mask)
        depth_should_be_joint = torch.where(mask_to_align_depth, depth_should_be, depth_should_be_ground)

        with torch.no_grad():
            depth_guide_joint, _ = kf_gen.get_depth(kf_gen.image_latest, target_depth=depth_should_be_joint, mask_align=joint_mask, archive_output=True, 
                                                    diffusion_steps=30, guidance_steps=8)

        kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy())

        kf_gen.image_latest = recomposed
        if config['gen_layer']:
            kf_gen.generate_layer(pred_semantic_map=sem_seg, scene_name=scene_name)

            depth_should_be = kf_gen.depth_latest_init
            mask_to_align_depth = ~(kf_gen.mask_disocclusion.bool()) & (depth_should_be < 0.006 * 0.8)
            mask_to_farther_depth = kf_gen.mask_disocclusion.bool() & (depth_should_be < 0.006 * 0.8)
            with torch.no_grad():
                kf_gen.depth, kf_gen.disparity = kf_gen.get_depth(kf_gen.image_latest, archive_output=True, target_depth=depth_should_be, mask_align=mask_to_align_depth, mask_farther=mask_to_farther_depth,
                                                                  diffusion_steps=30, guidance_steps=8)
            kf_gen.refine_disp_with_segments(no_refine_mask=ground_mask.squeeze().cpu().numpy(),
                                             existing_mask=~(kf_gen.mask_disocclusion).bool().squeeze().cpu().numpy(),
                                             existing_disp=kf_gen.disparity_latest_init.squeeze().cpu().numpy())
            wrong_depth_mask = kf_gen.depth_latest<kf_gen.depth_latest_init
            kf_gen.depth_latest[wrong_depth_mask] = kf_gen.depth_latest_init[wrong_depth_mask] + 0.0001
            kf_gen.depth_latest = kf_gen.mask_disocclusion * kf_gen.depth_latest + (1-kf_gen.mask_disocclusion) * kf_gen.depth_latest_init
            kf_gen.update_sky_mask()
            valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask)  # Base only
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest_init, depth=kf_gen.depth_latest_init, valid_mask=kf_gen.mask_disocclusion*outpaint_mask, gen_layer=True)  # Object layer
        else:
            valid_px_mask = outpaint_mask * (~kf_gen.sky_mask_latest)
            kf_gen.update_current_pc_by_kf(image=kf_gen.image_latest, depth=kf_gen.depth_latest, valid_mask=valid_px_mask)
        kf_gen.archive_latest()

        if config['gen_layer']:
            traindata, traindata_layer = kf_gen.convert_to_3dgs_traindata_latest_layer(xyz_scale=xyz_scale)
            gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
            scene = Scene(traindata_layer, gaussians, opt)
            dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
            save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene_layer{i+1:02d}"
            optimize_gaussians(gaussians, scene, opt, save_dir)  # Base layer optimization
        else:
            traindata = kf_gen.convert_to_3dgs_traindata_latest(xyz_scale=xyz_scale, use_no_loss_mask=False)

        if traindata['pcd_points'].shape[-1] == 0:
            gaussians.set_inscreen_points_to_visible(tdgs_cam)

            kf_gen.increment_kf_idx()
            keep_rendering = True
            continue
        
        mask_using_full_render = torch.zeros(1, 1, 512, 512).to(config['device'])
        x = torch.sum(fg_mask_0p5_nosky == 1, dim=2)>0  # [1, 512]
        x_idx = torch.nonzero(x, as_tuple=True)[1]
        if foreground_cols_idx.numel() > 0:
            min_index = foreground_cols_idx.min().item()
            max_index = foreground_cols_idx.max().item()
            mask_using_full_render[:, :, :x_idx.max().item(), min_index:max_index+1] = 1
        # mask_using_full_render[:, :, :sky_cond_width, :] = 1
        # mask_using_full_render[:, :, :side_sky_height, :sky_cond_width] = 1
        # mask_using_full_render[:, :, :side_sky_height, -sky_cond_width:] = 1
        
        mask_using_nosky_render = 1 - mask_using_full_render
        image_tmp = render_pkg_nosky["render"] * mask_using_nosky_render + render_pkg["render"] * mask_using_full_render
        
        
        gaussians = GaussianModel(sh_degree=0, previous_gaussian=gaussians)
        scene = Scene(traindata, gaussians, opt)
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        save_dir = Path(config['runs_dir']) / f"{dt_string}_gaussian_scene{i+1:02d}"
        optimize_gaussians(gaussians, scene, opt, save_dir)
        
        gaussians.set_inscreen_points_to_visible(tdgs_cam)

        kf_gen.increment_kf_idx()
        keep_rendering = True
        empty_cache()

def optimize_gaussians(gaussians: GaussianModel, scene: Scene, opt: GSParams, save_dir: Path, initialize_scaling=True):
    global latest_frame, iter_number, view_matrix, latest_viz
    iterable_gauss = range(1, opt.iterations + 1)
    trainCameras = scene.getTrainCameras().copy()
    gaussians.compute_3D_filter(cameras=trainCameras, initialize_scaling=initialize_scaling)

    for iteration in iterable_gauss:
        # Pick a random Camera
        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # import pdb; pdb.set_trace()
        # Render
        render_pkg = render(viewpoint_cam, gaussians, opt, background)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg['render'], render_pkg['viewspace_points'], render_pkg['visibility_filter'], render_pkg['radii'])

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()

        Ll1 = l1_loss(image, gt_image)
        
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if iteration == opt.iterations:
        # if iteration % 5 == 0 or iteration == 1:
            time.sleep(0.1)
            print(f'Iteration {iteration}, Loss: {loss.item()}')
            with torch.no_grad():
                tdgs_cam = convert_pt3d_cam_to_3dgs_cam(kf_gen.get_camera_by_js_view_matrix(view_matrix, xyz_scale=xyz_scale), xyz_scale=xyz_scale)
                render_pkg = render(tdgs_cam, gaussians, opt, background)
                image = render_pkg['render']
                # rendered_normal = render_pkg['render_normal']
                # rendered_normal_map = rendered_normal/2-0.5
            rendered_image = image.permute(1, 2, 0).detach().cpu().numpy()
            rendered_image = (rendered_image * 255).astype(np.uint8)
            rendered_image = rendered_image[..., ::-1]
            latest_frame = rendered_image
        loss.backward()
        if iteration == opt.iterations:
            print(f'Final loss: {loss.item()}')

        # Use variables that related to the trainable GS
        n_trainable = gaussians.get_xyz.shape[0]
        viewspace_point_tensor_grad, visibility_filter, radii = viewspace_point_tensor.grad[:n_trainable], visibility_filter[:n_trainable], radii[:n_trainable]

        with torch.no_grad():
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor_grad, visibility_filter)

                if iteration >= opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    max_screen_size = opt.max_screen_size if iteration >= opt.prune_from_iter else None
                    camera_height = 0.0003 * xyz_scale
                    scene_extent = camera_height * 2 if opt.scene_extent is None else opt.scene_extent
                    opacity_lowest = 0.05
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, opacity_lowest, scene_extent, max_screen_size)
                    gaussians.compute_3D_filter(cameras=trainCameras)
                
                # if (iteration % opt.opacity_reset_interval == 0 
                #     or (opt.white_background and iteration == opt.densify_from_iter)
                # ):
                #     gaussians.reset_opacity()

            # if iteration % 100 == 0 and iteration > opt.densify_until_iter:
            #     if iteration < opt.iterations - 100:
            #         # don't update in the end of training
            #         gaussians.compute_3D_filter(cameras=trainCameras)
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config"
    )
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    POSTMORTEM = config['debug']
    if POSTMORTEM:
        try:
            run(config)
        except Exception as e:
            print(e)
            import ipdb
            ipdb.post_mortem()
    else:
        run(config)
