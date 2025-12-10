# Density Based Revive Pruning train 코드.
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import json
import os
import torch
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from compute_scene_metrics import scene_metrics

def score_func(view, gaussians, pipeline, background, scores):

    img_scores = torch.zeros_like(scores)
    img_scores.requires_grad = True

    image = render(view, gaussians, pipeline, background,
                   scores=img_scores)['render']

    # Backward computes and stores grad squared values
    # in img_scores's grad
    image.sum().backward()

    scores += img_scores.grad


def spatial_density_revival(scores, prune_mask, xyz_positions, grid_size=32, revival_ratio=0.1):
    """
    공간 기반 그리드를 사용하여 밀집도가 높은 영역에서 제거된 가우시안들을 부활시킵니다.
    
    Args:
        scores: 전체 가우시안의 importance score 텐서
        prune_mask: 제거될 가우시안을 나타내는 boolean mask (True = 제거)
        xyz_positions: 가우시안들의 3D 위치 (N, 3)
        grid_size: 3D 그리드의 각 차원 크기
        revival_ratio: 밀집 영역에서 부활시킬 비율 (0.1 = 10%)
    
    Returns:
        updated_prune_mask: 부활 적용 후 업데이트된 prune mask
        revival_stats: 통계 정보 딕셔너리
    """
    
    # 제거될 가우시안들의 위치만 추출
    pruned_positions = xyz_positions[prune_mask]
    pruned_indices = torch.where(prune_mask)[0]
    
    if len(pruned_positions) == 0:
        return prune_mask, {"total_pruned": 0, "revived": 0, "high_density_cells": 0}
    
    # 3D 공간의 경계 계산
    min_coords = xyz_positions.min(dim=0)[0]
    max_coords = xyz_positions.max(dim=0)[0]
    
    # 각 차원의 범위
    ranges = max_coords - min_coords
    ranges = torch.clamp(ranges, min=1e-6) # 0으로 나누기 방지
    
    # 각 제거된 가우시안을 그리드 셀에 할당
    normalized_pos = (pruned_positions - min_coords) / ranges
    grid_indices = (normalized_pos * grid_size).long()
    grid_indices = torch.clamp(grid_indices, 0, grid_size - 1)
    
    # 각 셀을 고유 ID로 변환
    cell_ids = (grid_indices[:, 0] + 
                grid_indices[:, 1] * grid_size + 
                grid_indices[:, 2] * grid_size * grid_size)
    
    # 각 셀의 제거된 가우시안 개수 계산
    unique_cells, cell_counts = torch.unique(cell_ids, return_counts=True)
    
    # 밀집도 임계값 계산 (상위 20% 셀을 고밀도로 간주)
    if len(cell_counts) > 0:
        density_threshold = torch.quantile(cell_counts.float(), 0.8)
    else:
        density_threshold = 0
    
    high_density_cells = unique_cells[cell_counts >= density_threshold]
    
    # 부활시킬 가우시안 선택
    revival_mask = torch.zeros(len(pruned_indices), dtype=torch.bool, device=xyz_positions.device)
    
    for cell_id in high_density_cells:
        # 해당 셀에 속한 가우시안들의 인덱스
        cell_gaussian_mask = (cell_ids == cell_id)
        cell_gaussian_indices = torch.where(cell_gaussian_mask)[0]
        
        num_to_revive = max(1, int(len(cell_gaussian_indices) * revival_ratio))
        
        # 원본 인덱스에서 importance score를 직접 가져옴
        cell_original_indices = pruned_indices[cell_gaussian_indices]
        
        ### <<< 핵심 개선점: 인자로 받은 scores를 직접 사용 >>>
        cell_scores = scores[cell_original_indices]
        _, sorted_indices_in_cell = torch.sort(cell_scores, descending=True)
        selected_to_revive = sorted_indices_in_cell[:num_to_revive]
        
        # revival_mask는 pruned_indices 내에서의 상대적 위치이므로, selected_to_revive를 그대로 사용
        revival_mask[cell_gaussian_indices[selected_to_revive]] = True
    
    # 원본 prune_mask 업데이트 (부활시킬 것들은 False로 변경)
    updated_prune_mask = prune_mask.clone()
    updated_prune_mask[pruned_indices[revival_mask]] = False
    
    revival_stats = {
        "total_pruned": prune_mask.sum().item(),
        "revived": revival_mask.sum().item(),
        "high_density_cells": len(high_density_cells),
        "revival_ratio": revival_mask.sum().item() / max(1, prune_mask.sum().item()),
        "grid_size": grid_size,
        "density_threshold": density_threshold.item() if isinstance(density_threshold, torch.Tensor) else density_threshold
    }
    
    return updated_prune_mask, revival_stats


def prune(scene, gaussians, pipe, background, prune_ratio, use_spatial_revival=True, grid_size=32, revival_ratio=0.1):

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    torch.cuda.reset_peak_memory_stats()

    iter_start.record()

    with torch.enable_grad():
        pbar = tqdm(
            total=len(scene.getTrainCameras()),
            desc='Computing Pruning Scores')
        scores = torch.zeros_like(gaussians.get_opacity)
        for view in scene.getTrainCameras():
            score_func(view, gaussians, pipe, background,
                scores)
            pbar.update(1)
        pbar.close()


    
    # 원래의 prune 로직으로 prune mask 생성
    scores_squeezed = scores.squeeze()
    sorted_scores, sorted_indices = torch.sort(scores_squeezed, descending=False)
    num_to_prune = int(len(scores_squeezed) * prune_ratio)
    prune_mask = torch.zeros(len(scores_squeezed), dtype=torch.bool, device=scores.device)
    prune_mask[sorted_indices[:num_to_prune]] = True
    
    # Spatial density-based revival 적용
    revival_stats = None
    if use_spatial_revival and prune_mask.sum() > 0:
        xyz_positions = gaussians.get_xyz
        ### <<< 핵심 개선점: scores_squeezed를 직접 전달 >>>
        prune_mask, revival_stats = spatial_density_revival(
            scores_squeezed, prune_mask, xyz_positions, 
            grid_size=grid_size, revival_ratio=revival_ratio
        )
        
        print(f"\n[Spatial Revival] Revived {revival_stats['revived']}/{revival_stats['total_pruned']} gaussians "
              f"from {revival_stats['high_density_cells']} high-density cells")
    
    # 최종 prune mask로 가우시안 제거
    gaussians.prune_points(prune_mask)
    

    iter_end.record()

    # Track peak memory usage (in bytes) and convert to MB
    peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)
    time_ms = iter_start.elapsed_time(iter_end)
    time_min = time_ms / 60_000

    result = {
        "peak_memory_allocated" : peak_memory_allocated,
        "peak_memory_reserved" : peak_memory_reserved,
        "time_min" : time_min
    }
    
    if revival_stats:
        result.update(revival_stats)
    
    return result

def training(dataset, opt, pipe, testing_iterations, visualize_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    train_time_ms = 0
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    prune_time_min = 0
    prune_peak_memory_allocated = 0
    prune_peak_memory_reserved = 0
    
    # Spatial revival 통계 추적
    total_revived = 0
    revival_history = []

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        torch.cuda.reset_peak_memory_stats()
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()
        # Track peak memory usage (in bytes) and convert to MB
        peak_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_memory_reserved = torch.cuda.max_memory_reserved() / (1024 ** 2)


        with torch.no_grad():

            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                # --- Soft Pruning with Spatial Revival ---
                if (iteration >= opt.prune_from_iter) and \
                    (iteration < opt.prune_until_iter) and \
                    (iteration % opt.prune_interval == 0):
                    
                    gaussians.current_iteration = iteration
                    gaussians.model_path = scene.model_path
                    gaussians.prune_type = "soft"
                    
                    # Spatial revival parameters
                    use_revival = getattr(opt, 'use_spatial_revival', True)
                    grid_size = getattr(opt, 'spatial_grid_size', 32)
                    revival_ratio = getattr(opt, 'spatial_revival_ratio', 0.1)

                    prune_pkg = prune(
                        scene, gaussians, pipe, background,
                        0.85,
                        use_spatial_revival=use_revival,
                        grid_size=grid_size,
                        revival_ratio=revival_ratio)
                    
                    prune_time_min += prune_pkg['time_min']
                    prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
                    prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']
                    
                    # Revival 통계 저장
                    if 'revived' in prune_pkg:
                        total_revived += prune_pkg['revived']
                        revival_history.append({
                            'iteration': iteration,
                            'revived': prune_pkg['revived'],
                            'total_pruned': prune_pkg['total_pruned'],
                            'high_density_cells': prune_pkg['high_density_cells']
                        })

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            # --- Hard Pruning with Spatial Revival ---
            if (iteration >= opt.densify_until_iter) and \
                (iteration >= opt.prune_from_iter) and \
                (iteration < opt.prune_until_iter) and \
                (iteration % opt.prune_interval == 0):
                
                gaussians.current_iteration = iteration
                gaussians.model_path = scene.model_path
                gaussians.prune_type = "hard"
                
                # Spatial revival parameters
                use_revival = getattr(opt, 'use_spatial_revival', True)
                grid_size = getattr(opt, 'spatial_grid_size', 32)
                revival_ratio = getattr(opt, 'spatial_revival_ratio', 0.1)

                prune_pkg = prune(
                    scene, gaussians, pipe, background,
                    0.33,
                    use_spatial_revival=True,
                    grid_size=grid_size,
                    revival_ratio=revival_ratio)
                
                prune_time_min += prune_pkg['time_min']
                prune_peak_memory_allocated = prune_pkg['peak_memory_allocated']
                prune_peak_memory_reserved = prune_pkg['peak_memory_reserved']
                
                # Revival 통계 저장
                if 'revived' in prune_pkg:
                    total_revived += prune_pkg['revived']
                    revival_history.append({
                        'iteration': iteration,
                        'revived': prune_pkg['revived'],
                        'total_pruned': prune_pkg['total_pruned'],
                        'high_density_cells': prune_pkg['high_density_cells']
                    })


            # Log and save
            iter_time = iter_start.elapsed_time(iter_end)
            train_time_ms += iter_time
            train_time_min = train_time_ms / 60_000

            training_report(
                tb_writer, iteration, Ll1, loss,
                iter_time, train_time_min, prune_time_min,
                testing_iterations, visualize_iterations,
                peak_memory_allocated, peak_memory_reserved,
                prune_peak_memory_allocated, prune_peak_memory_reserved,
                scene, render,
                (pipe, background))
    
    # --- 추가된 부분: 최종 Importance Score 계산 및 저장 ---
    # in training() function, at the very end

    # --- 수정된 부분: 최종 가우시안 개수 출력 ---
    final_num_points = gaussians.get_xyz.shape[0]
    print(f"\n[INFO] Training complete. Final number of points: {final_num_points}")

    # ... (이하 Revival 통계 저장 부분은 그대로 둡니다) ...
        
    
    # Revival 통계 저장
    if revival_history:
        revival_output_path = os.path.join(scene.model_path, "spatial_revival_history.json")
        revival_summary = {
            "total_revived": total_revived,
            "num_revival_events": len(revival_history),
            "history": revival_history
        }
        with open(revival_output_path, 'w') as f:
            json.dump(revival_summary, f, indent=4)
        print(f"[INFO] Spatial revival history saved to {revival_output_path}")
        print(f"[INFO] Total gaussians revived: {total_revived}")

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(
        tb_writer, iteration, Ll1, loss,
        iter_time, train_time, prune_time,
        testing_iterations, visualize_iterations,
        peak_memory_allocated, peak_memory_reserved,
        prune_peak_memory_allocated, prune_peak_memory_reserved,
        scene : Scene, renderFunc, renderArgs):

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('time/iter_time', iter_time, iteration)
        tb_writer.add_scalar('time/train_time_minutes', train_time, iteration)
        tb_writer.add_scalar('time/prune_time_minutes', prune_time, iteration)
        tb_writer.add_scalar('counts/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('memory/peak_allocated_MB', peak_memory_allocated, iteration)
        tb_writer.add_scalar('memory/peak_reserved_MB', peak_memory_reserved, iteration)
        if prune_peak_memory_allocated > 0:
            tb_writer.add_scalar('memory/prune_peak_allocated_MB', prune_peak_memory_allocated, iteration)
        if prune_peak_memory_reserved > 0:
            tb_writer.add_scalar('memory/prune_peak_reserved_MB', prune_peak_memory_reserved, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        print("\n[ITER {}] Training Time: {} minutes".format(
            iteration, train_time))

        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()},
            {'name': 'train', 'cameras' : [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)] }
        )

        for config in validation_configs:
            name, cameras = config['name'], config['cameras']
            if cameras and len(cameras) > 0:
                metrics = scene_metrics(iteration, name, cameras,
                    scene, renderFunc, renderArgs)
                if tb_writer:
                    tb_writer.add_scalar(
                        f'metrics_{name}/L1 Loss', metrics[0], iteration)
                    tb_writer.add_scalar(
                        f'metrics_{name}/PSNR', metrics[1], iteration)
                    tb_writer.add_scalar(
                        f'metrics_{name}/SSIM', metrics[2], iteration)
                    tb_writer.add_scalar(
                        f'metrics_{name}/LPIPS', metrics[3], iteration)
                    tb_writer.add_scalar(
                        f'metrics_{name}/FPS', metrics[4], iteration)

    if (iteration in visualize_iterations) and tb_writer:
        validation_configs = (
            {'name': 'test', 'cameras' : scene.getTestCameras()},
            {'name': 'train', 'cameras' : [
                scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                for idx in range(5, 30, 5)] }
        )
        for config in validation_configs:
            for viewpoint in config['cameras'][:5]:
                image = torch.clamp(
                    renderFunc(
                        viewpoint, scene.gaussians, *renderArgs
                    )["render"],
                    0.0, 1.0)
                gt_image = torch.clamp(
                    viewpoint.original_image.to("cuda"),
                    0.0, 1.0)

                tb_writer.add_images(
                    config['name'] + "_view_{}/render".format(
                        viewpoint.image_name),
                    image[None], global_step=iteration)
                tb_writer.add_images(
                    config['name'] + "_view_{}/ground_truth".format(
                        viewpoint.image_name),
                    gt_image[None], global_step=iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--visualize_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    # Spatial Revival 관련 파라미터 추가
    parser.add_argument("--use_spatial_revival", action="store_true", default=True,
                        help="Enable spatial density-based revival mechanism")
    parser.add_argument("--spatial_grid_size", type=int, default=32,
                        help="Grid size for spatial density calculation")
    parser.add_argument("--spatial_revival_ratio", type=float, default=0.1,
                        help="Ratio of gaussians to revive in high-density regions (0.1 = 10%)")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    if args.use_spatial_revival:
        print(f"[Spatial Revival] Enabled with grid_size={args.spatial_grid_size}, revival_ratio={args.spatial_revival_ratio}")

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.visualize_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")