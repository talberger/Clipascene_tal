import os
import argparse
import subprocess as sp
from shutil import copyfile
import time


# ===========================================
# ======= fidelity axis generation ==========
# ===========================================
# This script is to run the baseline sketching method with the ViT clip model.
# We use this to create the first row in the abstraciton matrix.
# You can use this to create both the objects and background. 
# The default parameters are set for the background case.
# Example of a running command:
# CUDA_VISIBLE_DEVICES=1 python scripts/generate_fidelity_levels.py --im_name "man_flowers" --layer_opt 7 --object_or_background "object" --resize_obj 1
# CUDA_VISIBLE_DEVICES=3 python scripts/generate_fidelity_levels.py --im_name "man_flowers" --layer_opt 7 --object_or_background "background"
# CUDA_VISIBLE_DEVICES=2 python scripts/generate_fidelity_levels.py --im_name "man_flowers" --layer_opt 8 --object_or_background "background"


parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
parser.add_argument("--object_or_background", type=str, default="background")
parser.add_argument("--resize_obj", type=int, default=0)
parser.add_argument("--num_iter", type=int, default=1501)
# parser.add_argument("--num_iter", type=int, default=1501)
args = parser.parse_args()

#agruments for debug run
args.im_name = "ballerina"
args.layer_opt = [3, 5]
# args.object_or_background = "object"
# args.resize_obj = 1
args.num_iter = 1000
save_interval = 100
create_gif_bool = 0
path_svg="none"
if False: # Edit path svg
    init_layer = 8
    init_iter = 30
    init_by_best_iter = False

    if init_by_best_iter:
        path_svg =f"/home/SceneSketch/results_sketches/ballerina/runs/background_l{init_layer}_ballerina_mask/background_l{init_layer}_ballerina_mask_seed0_best.svg"
    else:
        path_svg =f"/home/SceneSketch/results_sketches/ballerina/runs/background_l{init_layer}_ballerina_mask/background_l{init_layer}_ballerina_mask_seed0/svg_logs/svg_iter{init_iter}.svg"


path_to_input_images = "./target_images" # where the input images are located
output_pref = f"./results_sketches/{args.im_name}/runs"

# if you run on objects, this need to be changed:
im_filename = f"{args.im_name}_mask.png"
folder_ = "background"
gradnorm = 0
mask_object = 0
if args.object_or_background == "object":
    if args.layer_opt != 4:
        gradnorm = 1
    mask_object = 1
    im_filename = f"{args.im_name}.png"
    folder_ = "scene"


# ===================
# ====== demo =======
# ===================
num_strokes = 64
num_sketches = 1 #changed for debug from 2 --> 1
num_iter = args.num_iter
# ===================


clip_conv_layer_weights_list = []
test_name_list = []
# set the weights for each layer
for l_opt in args.layer_opt:
    clip_conv_layer_weights_int = [0 for k in range(12)]
    if args.object_or_background == "object":
        # we combine two layers if we train on objects
        clip_conv_layer_weights_int[4] = 0.5
    clip_conv_layer_weights_int[l_opt] = 1
    clip_conv_layer_weights_str = [str(j) for j in clip_conv_layer_weights_int]
    clip_conv_layer_weights = ','.join(clip_conv_layer_weights_str)

    file_ = f"{path_to_input_images}/{folder_}/{im_filename}"
    test_name = f"{args.object_or_background}_l{l_opt}_{os.path.splitext(im_filename)[0]}"
    print(test_name)
    clip_conv_layer_weights_list.append(clip_conv_layer_weights)
    test_name_list.append(test_name)
start_time = time.time()
sp.run(["python", 
        "scripts/run_sketch_mp.py", 
        "--target_file", file_,
        "--output_pref", output_pref,
        "--num_iter", str(num_iter),
        "--test_name", test_name_list,
        "--num_sketches", str(num_sketches),
        "--mask_object", str(mask_object),
        "--clip_conv_layer_weights", clip_conv_layer_weights_list,
        "--gradnorm", str(gradnorm),
        "--resize_obj", str(args.resize_obj),
        "--eval_interval", str(50),
        "--min_eval_iter", str(400),
        "--save_interval", str(save_interval),
        "--path_svg", path_svg,
        "--create_gif_bool", str(create_gif_bool)])
total_time = time.time() - start_time
print(f"Time for one sketch fidelity [{total_time:.3f}] seconds")
