import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import math
import os
import sys
import re
import subprocess as sp
sys.stdout.flush()

import time
import traceback

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange

import config
import sketch_utils as utils
from models.loss import Loss
from models.painter_params import Painter, PainterOptimizer
from IPython.display import display, SVG
import matplotlib.pyplot as plt
from scripts.create_gif import run_create_gif
# from torch import autograd


def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_paths, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im,
                       mask=mask)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    masked_im, mask = utils.get_mask_u2net(args, target)
    
    if args.mask_object:
        target = masked_im
    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    transforms_.append(transforms.Resize(
        args.image_scale, interpolation=PIL.Image.BICUBIC))
    transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)

    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    mask = Image.fromarray((mask*255).astype(np.uint8)).convert('RGB')
    mask = data_transforms(mask).unsqueeze(0).to(args.device)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return target_, mask


def main(args):
    time_load_image_and_mask_start = time.time()
    inputs, mask = get_target(args)
    time_load_image_and_mask_end = time.time()
    time_load_image_and_mask = time_load_image_and_mask_end - time_load_image_and_mask_start
    # print(f"time_load_image_and_mask: {time_load_image_and_mask}")

    time_load_loss_start = time.time()
    loss_func = Loss(args, mask)
    time_load_loss_end =  time.time()
    time_load_loss= time_load_loss_end -  time_load_loss_start
    # print(f"time_load_loss_functions: {time_load_loss}")

 
    utils.log_input(args.use_wandb, 0, inputs, args.output_dir)
    renderer = load_renderer(args, inputs, mask)
    time_load_models_end = time.time()
    time_load_models = time_load_models_end - time_load_loss_end
    # print(f"time_load_models: {time_load_models}")
    
    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    best_loss, best_fc_loss, best_num_strokes = 100, 100, args.num_paths
    best_iter, best_iter_fc = 0, 0
    min_delta = 1e-7
    terminate = False

    time_init_storkes_start =  time.time()
    renderer.set_random_noise(0)
    renderer.init_image(stage=0)
    renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"init_svg") # this is the inital random strokes
    optimizer.init_optimizers()
    time_init_storkes_end =  time.time()
    time_init_storkes = time_init_storkes_end - time_init_storkes_start
    # print(f"time_init_storkes and init optimizer {time_init_storkes}")

    # not using tdqm for jupyter demo
    # print(f"number of iteration before change (epoch_range)= {args.num_iter}")
    
    if args.display:
        epoch_range = range(args.num_iter)
    else:
        epoch_range = tqdm(range(args.num_iter))

    if args.switch_loss:
        # start with width optim and than switch every switch_loss iterations
        renderer.turn_off_points_optim()
        optimizer.turn_off_points_optim()
    
    time_init_sketches_start = time.time()
    with torch.no_grad():
        img , _, _, _ = renderer.get_image("init")
        init_sketches = img.to(args.device)
        renderer.save_svg(
                f"{args.output_dir}", f"init")
    time_init_sketches = time.time() -time_init_sketches_start
    # print(f"time_init_sketches rendering and loading to device: {time_init_sketches}")
    # print(init_sketches.shape)

    #extract path svg det
    pattern_layer = r"_l(\d+)"
    pattern_iter  = r"_iter(\d+)"
    init_iter = 0
    match_layer = re.search(pattern_layer, args.path_svg)
    match_iter = re.search(pattern_iter, args.path_svg)
    if match_layer:
        init_layer = match_layer.group(1)
    if match_iter:
        init_iter = match_iter.group(1)
    init_title =''
    if args.path_svg != 'none':
        if "_best" in args.path_svg:
            init_title = f" initial sketch l{init_layer} best iteration"
        else:
            init_title = f" initial sketch l{init_layer} iter{init_iter}"

    
    time_get_image_accum = 0
    time_loss_calc_accum = 0
    time_opt_step_accum  = 0
    time_mlploc_loop_accum = 0
    time_mlpsimpe_loop_accum = 0
    time_rendring_one_sketch_accum = 0

    time_rendering_sketch_all_iteration_start = time.time()
    loss_list = []
    for epoch in epoch_range:
        if not args.display:
            epoch_range.refresh()
        start = time.time()
        optimizer.zero_grad_()
        # sketches ,time_mlploc_loop, time_mlpsimpe_loop, time_rendring_one_sketch = renderer.get_image().to(args.device)
        sketches ,time_mlploc_loop, time_mlpsimpe_loop, time_rendring_one_sketch = renderer.get_image()
        sketches.to(args.device)
        time_mlploc_loop_accum += time_mlploc_loop
        time_mlpsimpe_loop_accum += time_mlpsimpe_loop
        time_rendring_one_sketch_accum += time_rendring_one_sketch

        time_get_image_end = time.time()
        time_get_image = time_get_image_end - start
        time_get_image_accum += time_get_image
        # print(f"time_get_image: {time_get_image}")

        losses_dict_weighted, losses_dict_norm, losses_dict_original = loss_func(sketches, inputs.detach(), counter, renderer.get_widths(), renderer, optimizer, mode="train", width_opt=renderer.width_optim)
        loss = sum(list(losses_dict_weighted.values()))
        loss.backward()
        time_loss_calc_end = time.time()
        time_loss_calc = time_loss_calc_end - time_get_image_end
        time_loss_calc_accum += time_loss_calc
        # print(f"time_loss_calc: {time_loss_calc}")

        optimizer.step_()

        time_opt_step= time.time() -time_loss_calc_end
        time_opt_step_accum += time_opt_step
        # print(f"time_opt_step: {time_opt_step_accum}")

        if epoch % args.save_interval == 0:
            loss_list.append(loss.item())
            utils.plot_batch(init_sketches, inputs, sketches, len(epoch_range), init_title, f"{args.output_dir}/jpg_logs", counter, loss_list, args.save_interval 
                             ,use_wandb=args.use_wandb, title=f"iter{epoch}.jpg")
            renderer.save_svg(
                f"{args.output_dir}/svg_logs", f"svg_iter{epoch}")

        if epoch % args.eval_interval == 0 and epoch >= args.min_eval_iter:
            if args.width_optim:
                if args.mlp_train and args.optimize_points:
                    torch.save({
                        'model_state_dict': renderer.get_mlp().state_dict(),
                        'optimizer_state_dict': optimizer.get_points_optim().state_dict(),
                        }, f"{args.output_dir}/mlps/points_mlp{counter}.pt")
                torch.save({
                    'model_state_dict': renderer.get_width_mlp().state_dict(),
                    'optimizer_state_dict': optimizer.get_width_optim().state_dict(),
                    }, f"{args.output_dir}/mlps/width_mlp{counter}.pt")

            with torch.no_grad():
                losses_dict_weighted_eval, losses_dict_norm_eval, losses_dict_original_eval = loss_func(sketches, inputs, counter, renderer.get_widths(), renderer=renderer, mode="eval", width_opt=renderer.width_optim)
                loss_eval = sum(list(losses_dict_weighted_eval.values()))
                configs_to_save["loss_eval"].append(loss_eval.item())
                if "num_strokes" not in configs_to_save.keys():
                    configs_to_save["num_strokes"] = []
                configs_to_save["num_strokes"].append(renderer.get_strokes_count())
                for k in losses_dict_norm_eval.keys():
                    original_name, gradnorm_name, final_name = k + "_original_eval", k + "_gradnorm_eval", k + "_final_eval"
                    if original_name not in configs_to_save.keys():
                        configs_to_save[original_name] = []
                    if gradnorm_name not in configs_to_save.keys():
                        configs_to_save[gradnorm_name] = []
                    if final_name not in configs_to_save.keys():
                        configs_to_save[final_name] = []
                    
                    configs_to_save[original_name].append(losses_dict_original_eval[k].item())
                    configs_to_save[gradnorm_name].append(losses_dict_norm_eval[k].item())
                    if k in losses_dict_weighted_eval.keys():
                        configs_to_save[final_name].append(losses_dict_weighted_eval[k].item())                

                cur_delta = loss_eval.item() - best_loss
                if abs(cur_delta) > min_delta:
                    if cur_delta < 0:
                        best_loss = loss_eval.item()
                        best_iter = epoch
                        best_num_strokes = renderer.get_strokes_count()
                        terminate = False
                        
                        if args.mlp_train and args.optimize_points and not args.width_optim:
                            torch.save({
                                'model_state_dict': renderer.get_mlp().state_dict(),
                                'optimizer_state_dict': optimizer.get_points_optim().state_dict(),
                                }, f"{args.output_dir}/points_mlp.pt")
                        
                if args.use_wandb:
                    wandb.run.summary["best_loss"] = best_loss
                    wandb.run.summary["best_loss_fc"] = best_fc_loss
                    wandb.run.summary["best_num_strokes"] = best_num_strokes
                    wandb_dict = {"delta": cur_delta,
                                  "loss_eval": loss_eval.item()}
                    for k in losses_dict_original_eval.keys():
                        wandb_dict[k + "_original_eval"] = losses_dict_original_eval[k].item()                    
                    for k in losses_dict_norm_eval.keys():
                        wandb_dict[k + "_gradnorm_eval"] = losses_dict_norm_eval[k].item()
                    for k in losses_dict_weighted_eval.keys():
                        wandb_dict[k + "_final_eval"] = losses_dict_weighted_eval[k].item()
                    wandb.log(wandb_dict, step=counter)
        if counter == 0 and args.attention_init:
            utils.plot_atten(renderer.get_attn(), renderer.get_thresh(), inputs, renderer.get_inds(),
                             args.use_wandb, "{}/{}.jpg".format(
                                 args.output_dir, "attention_map"),
                             args.saliency_model, args.display_logs)

        if args.use_wandb:
            wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr()}
            if args.width_optim:
                wandb_dict["lr_width"] = optimizer.get_lr("width")
                wandb_dict["num_strokes"] = renderer.get_strokes_count()
            # wandb_dict = {"loss": loss.item(), "lr": optimizer.get_lr(), "num_strokes": optimizer.}
            for k in losses_dict_original.keys():
                wandb_dict[k + "_original"] = losses_dict_original[k].item()
            for k in losses_dict_norm.keys():
                wandb_dict[k + "_gradnorm"] = losses_dict_norm[k].item()
            for k in losses_dict_weighted.keys():
                wandb_dict[k + "_final"] = losses_dict_weighted[k].item()
            
            wandb.log(wandb_dict, step=counter)

        counter += 1
        if args.switch_loss:
            if epoch > 0 and epoch % args.switch_loss == 0:
                    renderer.switch_opt()
                    optimizer.switch_opt()
    if args.width_optim:
        utils.log_best_normalised_sketch(configs_to_save, args.output_dir, args.use_wandb, args.device, args.eval_interval, args.min_eval_iter)
    utils.inference_sketch(args)

    time_rendering_sketch_all_iteration_end =  time.time()
    time_rendering_sketch_all_iteration =  time_rendering_sketch_all_iteration_end - time_rendering_sketch_all_iteration_start

    if args.create_gif_bool:
        # title_prefix = output_dir.split('/')[-2].replace('_', ' ')
        # sp.run(['python', '/home/SceneSketch/scripts/create_gif.py',"aaa", "bbb"])
        #sp.run(['python', '/home/SceneSketch/scripts/create_gif.py',f"{args.output_dir}/jpg_logs/*.jpg", "/home/SceneSketch/results_sketches/anim/"+"anim_" +args.output_dir.split('/')[-1]+".gif"  ])
        # scripts.create_gif.main(f"{args.output_dir}/jpg_logs/*.jpg","/home/SceneSketch/results_sketches/anim/"+"anim_" +args.output_dir.split('/')[-1]+".gif" )
        run_create_gif(f"{args.output_dir}/jpg_logs/*.jpg", "/home/SceneSketch/results_sketches/anim/"+"anim_" +args.output_dir.split('/')[-1]+init_title +".gif")
        # run create_gif(f"{args.output_dir}/jpg_logs/*.gpj", "/home/SceneSketch/results_sketches/anim/"+args.output_dir.split('/')[-1]+".jpg")
    

    print(f"############# times #################")
    print(f"time_load_image_and_mask: {time_load_image_and_mask}")
    print(f"time_load_loss_functions: {time_load_loss}")
    print(f"time_load_models: {time_load_models}")
    print(f"time_init_storkes and init optimizer {time_init_storkes}")
    print(f"time_init_sketches rendering and loading to device: {time_init_sketches}")
    print(f'time_rendering_sketch_all_iteration {time_rendering_sketch_all_iteration}')


    # times per iteration
    # print(f"time_get_image: {time_get_image_accum*1000/len(epoch_range)}")
    # print(f"    time_mlploc_loop_accum: {time_mlploc_loop_accum*1000/len(epoch_range)}")
    # print(f"    time_mlpsimpe_loop_accum: {time_mlpsimpe_loop_accum*1000/len(epoch_range)}")
    # print(f"    time_rendring_one_sketch: {time_rendring_one_sketch_accum*1000/len(epoch_range)}")
    # print(f"time_loss_calc_accum: {time_loss_calc_accum*1000/len(epoch_range)}")
    # print(f"time_opt_step_accum: {time_opt_step_accum*1000/len(epoch_range)}")

    return configs_to_save

if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()
