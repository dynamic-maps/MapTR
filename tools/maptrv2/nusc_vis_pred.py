import argparse
import mmcv
import os
import shutil
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, scatter_kwargs
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet3d.utils import collect_env, get_root_logger
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
import sys
sys.path.append('')
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import numpy as np

# torch>=2.6 defaults torch.load(weights_only=True), which breaks loading
# legacy checkpoints (e.g. containing numpy scalars) via mmcv
_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import transforms
from matplotlib.patches import Rectangle
import cv2


CAMS = ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT',]
# we choose these samples not because it is easy but because it is hard
CANDIDATE=['n008-2018-08-01-15-16-36-0400_1533151184047036',
           'n008-2018-08-01-15-16-36-0400_1533151200646853',
           'n008-2018-08-01-15-16-36-0400_1533151274047332',
           'n008-2018-08-01-15-16-36-0400_1533151369947807',
           'n008-2018-08-01-15-16-36-0400_1533151581047647',
           'n008-2018-08-01-15-16-36-0400_1533151585447531',
           'n008-2018-08-01-15-16-36-0400_1533151741547700',
           'n008-2018-08-01-15-16-36-0400_1533151854947676',
           'n008-2018-08-22-15-53-49-0400_1534968048946931',
           'n008-2018-08-22-15-53-49-0400_1534968255947662',
           'n008-2018-08-01-15-16-36-0400_1533151616447606',
           'n015-2018-07-18-11-41-49+0800_1531885617949602',
           'n008-2018-08-28-16-43-51-0400_1535489136547616',
           'n008-2018-08-28-16-43-51-0400_1535489145446939',
           'n008-2018-08-28-16-43-51-0400_1535489152948944',
           'n008-2018-08-28-16-43-51-0400_1535489299547057',
           'n008-2018-08-28-16-43-51-0400_1535489317946828',
           'n008-2018-09-18-15-12-01-0400_1537298038950431',
           'n008-2018-09-18-15-12-01-0400_1537298047650680',
           'n008-2018-09-18-15-12-01-0400_1537298056450495',
           'n008-2018-09-18-15-12-01-0400_1537298074700410',
           'n008-2018-09-18-15-12-01-0400_1537298088148941',
           'n008-2018-09-18-15-12-01-0400_1537298101700395',
           'n015-2018-11-21-19-21-35+0800_1542799330198603',
           'n015-2018-11-21-19-21-35+0800_1542799345696426',
           'n015-2018-11-21-19-21-35+0800_1542799353697765',
           'n015-2018-11-21-19-21-35+0800_1542799525447813',
           'n015-2018-11-21-19-21-35+0800_1542799676697935',
           'n015-2018-11-21-19-21-35+0800_1542799758948001',
           ]

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def colors_plt_to_bgr(colors_plt):
    """Convert matplotlib color names to OpenCV BGR uint8 tuples."""
    bgr = []
    for name in colors_plt:
        r, g, b = mcolors.to_rgb(name)
        bgr.append((int(b * 255), int(g * 255), int(r * 255)))
    return bgr


def denorm_cam_img(img_tensor, mean, std, to_bgr):
    """Undo Normalize() on a single (3, H, W) camera image tensor for cv2 drawing."""
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    if to_bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(img_np)


def get_ground_z(img_meta, fallback=-1.6):
    """Height of the ground plane in the lidar frame for this sample.

    The lidar sits above the vehicle's ground reference by a calibrated
    amount (varies per vehicle/sensor rig), so it must be read from
    'lidar2ego' per-sample rather than assumed constant across scenes.
    Falls back to the repo's existing -1.6 default (see
    nuscenes_offlinemap_dataset.line_ego_to_pvmask) if unavailable.
    """
    lidar2ego = img_meta.get('lidar2ego')
    if lidar2ego is None:
        return fallback
    return -float(np.asarray(lidar2ego)[2, 3])


def add_bev_legend(colors_plt, class_names):
    """Add a class-name/color legend to the current matplotlib BEV plot."""
    handles = [plt.Line2D([0], [0], color=c, lw=2) for c in colors_plt]
    plt.legend(handles, class_names, loc='upper right', fontsize=4,
              framealpha=0.6, handlelength=1.2, borderpad=0.3, labelspacing=0.3)


def draw_lines_on_cam_img(cam_img, lines_xy, labels, lidar2img_mat, colors_bgr, thickness=2,
                          ground_z=-1.6):
    """Project ground-plane map lines into a camera image and draw them.

    lines_xy: list of (N, 2) arrays in lidar/ego coordinates.
    lidar2img_mat: (4, 4) matrix mapping lidar coords to this camera's pixels.
    ground_z: z of the road surface in the lidar frame (lidar sits above the
        ground, so the ground is at a negative z, not z=0). Compute this
        per-sample with get_ground_z() rather than hardcoding, since the
        lidar mounting height varies by vehicle/sensor rig.

    NOTE: this assumes a perfectly flat, level ground (no pitch/roll
    correction). An attempt to correct for road slope/vehicle pitch via
    lidar2global's full rotation made alignment worse (front/back cameras
    skewed in opposite directions), so it was reverted; sloped-road scenes
    may still show some offset, especially at range.
    """
    for pts, label in zip(lines_xy, labels):
        proj_pts = []
        for x, y in pts:
            p = lidar2img_mat @ np.array([x, y, ground_z, 1.0])
            if p[2] <= 1e-3:
                proj_pts.append(None)
                continue
            u, v = p[0] / p[2], p[1] / p[2]
            proj_pts.append((int(np.clip(u, -1e5, 1e5)), int(np.clip(v, -1e5, 1e5))))
        color = colors_bgr[int(label)]
        for p1, p2 in zip(proj_pts[:-1], proj_pts[1:]):
            if p1 is None or p2 is None:
                continue
            cv2.line(cam_img, p1, p2, color, thickness, cv2.LINE_AA)
    return cam_img


def draw_legend_cv2(img, class_names, colors_bgr, origin=(10, 10)):
    """Draw a color-swatch legend (class name + line color) in the top-left corner."""
    x0, y0 = origin
    row_h = 22
    swatch_w = 24
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    text_sizes = [cv2.getTextSize(name, font, font_scale, thickness)[0] for name in class_names]
    box_w = swatch_w + 8 + max(w for w, h in text_sizes) + 10
    box_h = row_h * len(class_names) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    for i, (name, color) in enumerate(zip(class_names, colors_bgr)):
        cy = y0 + 5 + i * row_h
        cv2.line(img, (x0 + 5, cy + row_h // 2), (x0 + 5 + swatch_w, cy + row_h // 2), color, 3, cv2.LINE_AA)
        cv2.putText(img, name, (x0 + swatch_w + 13, cy + row_h // 2 + 5),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def parse_args():
    parser = argparse.ArgumentParser(description='vis hdmaptr map gt label')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--score-thresh', default=0.4, type=float, help='samples to visualize')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument('--show-cam', action='store_true', help='show camera pic')
    parser.add_argument(
        '--gt-format',
        type=str,
        nargs='+',
        default=['fixed_num_pts',],
        help='vis format, default should be "points",'
        'support ["se_pts","bbox","fixed_num_pts","polyline_pts"]')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.show_dir is None:
        args.show_dir = osp.join('./work_dirs', 
                                osp.splitext(osp.basename(args.config))[0],
                                'vis_pred')
    # create vis_label dir
    mmcv.mkdir_or_exist(osp.abspath(args.show_dir))
    cfg.dump(osp.join(args.show_dir, osp.basename(args.config)))
    logger = get_root_logger()
    logger.info(f'DONE create vis_pred dir: {args.show_dir}')


    dataset = build_dataset(cfg.data.test)
    dataset.is_vis_on_test = True #TODO, this is a hack
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        # workers_per_gpu=cfg.data.workers_per_gpu,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    logger.info('Done build test data set')

    # build the model and load checkpoint
    # import pdb;pdb.set_trace()
    cfg.model.train_cfg = None
    # cfg.model.pts_bbox_head.bbox_coder.max_num=15 # TODO this is a hack
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    logger.info('loading check point')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    logger.info('DONE load check point')
    # NOTE: MMDataParallel's scatter path passes raw int device ids into
    # torch internals that now expect torch.device objects; bypass it and
    # unwrap DataContainer manually before calling the model instead.
    model = model.cuda()
    model.eval()

    img_norm_cfg = cfg.img_norm_cfg

    # get denormalized param
    mean = np.array(img_norm_cfg['mean'],dtype=np.float32)
    std = np.array(img_norm_cfg['std'],dtype=np.float32)
    to_bgr = img_norm_cfg['to_rgb']

    # get pc_range
    pc_range = cfg.point_cloud_range

    # get car icon
    car_img = Image.open('./figs/lidar_car.png')

    # get color map: divider->r, ped->b, boundary->g
    colors_plt = ['orange', 'b', 'r', 'g']
    colors_bgr = colors_plt_to_bgr(colors_plt)

    logger.info('BEGIN vis test dataset samples gt label & pred')

    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    legend_class_names = list(dataset.MAPCLASSES)
    have_mask = False
    # prog_bar = mmcv.ProgressBar(len(CANDIDATE))
    prog_bar = mmcv.ProgressBar(len(dataset))
    # import pdb;pdb.set_trace()
    for i, data in enumerate(data_loader):
        if ~(data['gt_labels_3d'].data[0][0] != -1).any():
            # import pdb;pdb.set_trace()
            logger.error(f'\n empty gt for index {i}, continue')
            # prog_bar.update()  
            continue
       
        
        img = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        gt_bboxes_3d = data['gt_bboxes_3d'].data[0]
        gt_labels_3d = data['gt_labels_3d'].data[0]

        pts_filename = img_metas[0]['pts_filename']
        pts_filename = osp.basename(pts_filename)
        pts_filename = pts_filename.replace('__LIDAR_TOP__', '_').split('.')[0]
        # import pdb;pdb.set_trace()
        # if pts_filename not in CANDIDATE:
        #     continue

        with torch.no_grad():
            _, kwargs = scatter_kwargs(None, data, [next(model.parameters()).device])
            result = model(return_loss=False, rescale=True, **kwargs[0])
        sample_dir = osp.join(args.show_dir, pts_filename)
        mmcv.mkdir_or_exist(osp.abspath(sample_dir))

        filename_list = img_metas[0]['filename']
        img_path_dict = {}
        # save cam img for sample
        for filepath in filename_list:
            filename = osp.basename(filepath)
            filename_splits = filename.split('__')
            # sample_dir = filename_splits[0]
            # sample_dir = osp.join(args.show_dir, sample_dir)
            # mmcv.mkdir_or_exist(osp.abspath(sample_dir))
            img_name = filename_splits[1] + '.jpg'
            img_path = osp.join(sample_dir,img_name)
            # img_path_list.append(img_path)
            shutil.copyfile(filepath,img_path)
            img_path_dict[filename_splits[1]] = img_path
         
        # surrounding view
        row_1_list = []
        for cam in CAMS[:3]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_1_list.append(cam_img)
        row_2_list = []
        for cam in CAMS[3:]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_2_list.append(cam_img)
        row_1_img=cv2.hconcat(row_1_list)
        row_2_img=cv2.hconcat(row_2_list)
        cams_img = cv2.vconcat([row_1_img,row_2_img])
        cams_img_path = osp.join(sample_dir,'surroud_view.jpg')
        cv2.imwrite(cams_img_path, cams_img,[cv2.IMWRITE_JPEG_QUALITY, 70])
        
        for vis_format in args.gt_format:
            if vis_format == 'se_pts':
                gt_line_points = gt_bboxes_3d[0].start_end_points
                for gt_bbox_3d, gt_label_3d in zip(gt_line_points, gt_labels_3d[0]):
                    pts = gt_bbox_3d.reshape(-1,2).numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])
            elif vis_format == 'bbox':
                gt_lines_bbox = gt_bboxes_3d[0].bbox
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_bbox, gt_labels_3d[0]):
                    gt_bbox_3d = gt_bbox_3d.numpy()
                    xy = (gt_bbox_3d[0],gt_bbox_3d[1])
                    width = gt_bbox_3d[2] - gt_bbox_3d[0]
                    height = gt_bbox_3d[3] - gt_bbox_3d[1]
                    # import pdb;pdb.set_trace()
                    plt.gca().add_patch(Rectangle(xy,width,height,linewidth=0.4,edgecolor=colors_plt[gt_label_3d],facecolor='none'))
                    # plt.Rectangle(xy, width, height,color=colors_plt[gt_label_3d])
                # continue
            elif vis_format == 'fixed_num_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                # gt_bboxes_3d[0].fixed_num=30 #TODO, this is a hack
                gt_lines_fixed_num_pts = gt_bboxes_3d[0].fixed_num_sampled_points
                for gt_bbox_3d, gt_label_3d in zip(gt_lines_fixed_num_pts, gt_labels_3d[0]):
                    # import pdb;pdb.set_trace() 
                    pts = gt_bbox_3d.numpy()
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                    
                    plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=2,alpha=0.8,zorder=-1)
                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    # plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
                add_bev_legend(colors_plt, legend_class_names)

                gt_fixedpts_map_path = osp.join(sample_dir, 'GT_fixednum_pts_MAP.png')
                plt.savefig(gt_fixedpts_map_path, bbox_inches='tight', format='png',dpi=1200)
                plt.close()   
            elif vis_format == 'polyline_pts':
                plt.figure(figsize=(2, 4))
                plt.xlim(pc_range[0], pc_range[3])
                plt.ylim(pc_range[1], pc_range[4])
                plt.axis('off')
                gt_lines_instance = gt_bboxes_3d[0].instance_list
                # import pdb;pdb.set_trace()
                for gt_line_instance, gt_label_3d in zip(gt_lines_instance, gt_labels_3d[0]):
                    pts = np.array(list(gt_line_instance.coords))
                    x = np.array([pt[0] for pt in pts])
                    y = np.array([pt[1] for pt in pts])
                    
                    # plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1, color=colors_plt[gt_label_3d])

                    # plt.plot(x, y, color=colors_plt[gt_label_3d])
                    plt.plot(x, y, color=colors_plt[gt_label_3d],linewidth=1,alpha=0.8,zorder=-1)
                    plt.scatter(x, y, color=colors_plt[gt_label_3d],s=1,alpha=0.8,zorder=-1)
                plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])

                gt_polyline_map_path = osp.join(sample_dir, 'GT_polyline_pts_MAP.png')
                plt.savefig(gt_polyline_map_path, bbox_inches='tight', format='png',dpi=1200)
                plt.close()           

            else: 
                logger.error(f'WRONG visformat for GT: {vis_format}')
                raise ValueError(f'WRONG visformat for GT: {vis_format}')


        # import pdb;pdb.set_trace()
        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')

        # visualize pred
        # import pdb;pdb.set_trace()
        result_dic = result[0]['pts_bbox']
        boxes_3d = result_dic['boxes_3d'] # bbox: xmin, ymin, xmax, ymax
        scores_3d = result_dic['scores_3d']
        labels_3d = result_dic['labels_3d']
        pts_3d = result_dic['pts_3d']
        keep = scores_3d > args.score_thresh

        # project predicted map lines onto each camera image (ground plane)
        pred_lines_xy = [pred_pts.numpy() for pred_pts in pts_3d[keep]]
        pred_labels_for_cam = labels_3d[keep].tolist()
        lidar2img_list = img_metas[0]['lidar2img']
        ground_z = get_ground_z(img_metas[0])
        # NOTE: img[0, cam_idx]'s camera order follows filename_list (as set
        # by the dataset's camera_types order), which does NOT match CAMS
        # (front_left/front/front_right/...); derive the name per index
        # instead of assuming enumerate(CAMS) lines up with the tensor.
        cam_pred_imgs_by_name = {}
        for cam_idx, filepath in enumerate(filename_list):
            cam_name = osp.basename(filepath).split('__')[1]
            cam_img = denorm_cam_img(img[0, cam_idx], mean, std, to_bgr)
            lidar2img_mat = np.array(lidar2img_list[cam_idx])
            cam_img = draw_lines_on_cam_img(
                cam_img, pred_lines_xy, pred_labels_for_cam, lidar2img_mat, colors_bgr,
                ground_z=ground_z)
            cam_img = draw_legend_cv2(cam_img, legend_class_names, colors_bgr)
            cv2.imwrite(osp.join(sample_dir, cam_name + '_PRED.jpg'), cam_img,
                       [cv2.IMWRITE_JPEG_QUALITY, 70])
            cam_pred_imgs_by_name[cam_name] = cam_img
        cam_pred_imgs = [cam_pred_imgs_by_name[cam_name] for cam_name in CAMS]
        row_1_pred = cv2.hconcat(cam_pred_imgs[:3])
        row_2_pred = cv2.hconcat(cam_pred_imgs[3:])
        surround_pred_img = cv2.vconcat([row_1_pred, row_2_pred])
        cv2.imwrite(osp.join(sample_dir, 'surroud_view_PRED.jpg'), surround_pred_img,
                   [cv2.IMWRITE_JPEG_QUALITY, 70])

        # DEBUG: also project GT lines the same way, to isolate whether any
        # misalignment comes from the projection math or from model error.
        gt_lines_xy = [gt_pts.numpy() for gt_pts in gt_bboxes_3d[0].fixed_num_sampled_points]
        gt_labels_for_cam = gt_labels_3d[0].tolist()
        cam_gt_imgs_by_name = {}
        for cam_idx, filepath in enumerate(filename_list):
            cam_name = osp.basename(filepath).split('__')[1]
            cam_img = denorm_cam_img(img[0, cam_idx], mean, std, to_bgr)
            lidar2img_mat = np.array(lidar2img_list[cam_idx])
            cam_img = draw_lines_on_cam_img(
                cam_img, gt_lines_xy, gt_labels_for_cam, lidar2img_mat, colors_bgr,
                ground_z=ground_z)
            cam_img = draw_legend_cv2(cam_img, legend_class_names, colors_bgr)
            cv2.imwrite(osp.join(sample_dir, cam_name + '_GTPROJ.jpg'), cam_img,
                       [cv2.IMWRITE_JPEG_QUALITY, 70])
            cam_gt_imgs_by_name[cam_name] = cam_img
        cam_gt_imgs = [cam_gt_imgs_by_name[cam_name] for cam_name in CAMS]
        row_1_gt = cv2.hconcat(cam_gt_imgs[:3])
        row_2_gt = cv2.hconcat(cam_gt_imgs[3:])
        surround_gt_img = cv2.vconcat([row_1_gt, row_2_gt])
        cv2.imwrite(osp.join(sample_dir, 'surroud_view_GTPROJ.jpg'), surround_gt_img,
                   [cv2.IMWRITE_JPEG_QUALITY, 70])

        plt.figure(figsize=(2, 4))
        plt.xlim(pc_range[0], pc_range[3])
        plt.ylim(pc_range[1], pc_range[4])
        plt.axis('off')
        for pred_score_3d, pred_bbox_3d, pred_label_3d, pred_pts_3d in zip(scores_3d[keep], boxes_3d[keep],labels_3d[keep], pts_3d[keep]):

            pred_pts_3d = pred_pts_3d.numpy()
            pts_x = pred_pts_3d[:,0]
            pts_y = pred_pts_3d[:,1]
            plt.plot(pts_x, pts_y, color=colors_plt[pred_label_3d],linewidth=1,alpha=0.8,zorder=-1)
            plt.scatter(pts_x, pts_y, color=colors_plt[pred_label_3d],s=1,alpha=0.8,zorder=-1)


            pred_bbox_3d = pred_bbox_3d.numpy()
            xy = (pred_bbox_3d[0],pred_bbox_3d[1])
            width = pred_bbox_3d[2] - pred_bbox_3d[0]
            height = pred_bbox_3d[3] - pred_bbox_3d[1]
            pred_score_3d = float(pred_score_3d)
            pred_score_3d = round(pred_score_3d, 2)
            s = str(pred_score_3d)

        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        add_bev_legend(colors_plt, legend_class_names)

        map_path = osp.join(sample_dir, 'PRED_MAP_plot.png')
        plt.savefig(map_path, bbox_inches='tight', format='png',dpi=1200)
        plt.close()

        prog_bar.update()

    logger.info('\n DONE vis test dataset samples gt label & pred')
if __name__ == '__main__':
    main()
