from PIL import Image, ImageSequence
import glob
import re

# path_to_jpg_files = "/home/SceneSketch/results_sketches/bull/runs/background_l4_bull_mask/background_l4_bull_mask_seed0/jpg_logs/*.jpg"
# path_to_jpg_files = "/home/SceneSketch/results_sketches/bull/runs/object_l4_bull/object_l4_bull_seed0/jpg_logs/*.jpg"
# path_to_jpg_files = "/home/SceneSketch/results_sketches/ballerina/runs/background_l4_ballerina_mask/background_l4_ballerina_mask_seed0/jpg_logs/*.jpg"
# path_to_jpg_files = "/home/SceneSketch/results_sketches/ballerina/runs/object_l4_ballerina/object_l4_ballerina_seed0/jpg_logs/*.jpg"
# path_to_jpg_files = "/home/SceneSketch/results_sketches/ballerina/runs/background_l2_ballerina_mask/background_l2_ballerina_mask_seed0/jpg_logs/*.jpg"
# path_to_jpg_files = "/home/SceneSketch/results_sketches/ballerina/runs/background_l8_ballerina_mask/background_l8_ballerina_mask_seed0/jpg_logs/*.jpg"
# path_to_jpg_files = "/home/SceneSketch/results_sketches/ballerina/runs/background_l4_ballerina_mask/background_l4_ballerina_mask_seed0/jpg_logs/*.jpg"
# path_to_jpg_files = "/home/SceneSketch/results_sketches/ballerina/runs/object_l2_ballerina/object_l2_ballerina_seed0/jpg_logs/*.jpg"
# path_to_jpg_files = "/home/SceneSketch/results_sketches/ballerina/runs/object_l8_ballerina/object_l8_ballerina_seed0/jpg_logs/*.jpg"
# path_to_jpg_files = "/home/SceneSketch/results_sketches/ballerina/runs/object_l11_ballerina/object_l11_ballerina_seed0/jpg_logs/*.jpg"

# path_to_save_anim = "/home/SceneSketch/results_sketches/anim/ballerina_bg_l4_anim_init_L2_iter100.gif"

# Define a function to pad the numeric part with zeros
def pad_with_zeros(match):
    return match.group(1) + '{:04d}'.format(int(match.group(2)))

def remove_leading_zeros(match):
    return match.group(1) + str(int(match.group(2)))


def run_create_gif(path_to_jpg_files,path_to_save_anim):

    print(f"path_to_jpg_files: {path_to_jpg_files}")
    print(f"path_to_save_anim: {path_to_save_anim}")


    # List all JPG files in your directory
    jpg_files = glob.glob(path_to_jpg_files)

    padded_file_names = []
    for file_name in jpg_files:
        new_file_name = re.sub(r'(iter)(\d+)', pad_with_zeros, file_name)
        padded_file_names.append(new_file_name)

    padded_file_names_sorted = sorted(padded_file_names)

    jpg_files_sorted = []
    for file_name in padded_file_names_sorted:
        new_file_name = re.sub(r'(iter)(\d+)', remove_leading_zeros, file_name)
        jpg_files_sorted.append(new_file_name)
    # Create a list to hold image frames
    frames = []

    # Load each JPG file and append it to the frames list
    for jpg_file in jpg_files_sorted:
        img = Image.open(jpg_file)
        frames.append(img)

    # Create a GIF file from the frames
    gif_file = path_to_save_anim
    frames[0].save(
        gif_file,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # Set the duration between frames in milliseconds (adjust as needed)
        loop=0  # 0 means infinite loop, you can set a different number for a finite loop
    )

    print(f"GIF animation saved to {gif_file}")
