import subprocess as sp
import argparse
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4,5,6,7'
import multiprocessing as mp
import logging

# Set up logging to file
# if os.path.exists('results_sketches/log_file_main.log'):
#     os.remove('results_sketches/log_file_main.log')
# if os.path.exists('results_sketches/log_file_fidelity.log'):
#     os.remove('results_sketches/log_file_fidelity.log')
# if os.path.exists('results_sketches/log_file_simplicity.log'):
#     os.remove('results_sketches/log_file_simplicity.log')
logging.basicConfig(filename='results_sketches/log_file_main.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# ===================================================
# ================= run background ==================
# ===================================================
# The script generates a 2x2 matrix of different levels of fidelity and simplicity. 
# You can specify different layers and divs using the --layers and
# --divs parameters.
# Example of running commands:
# python scripts/run_4_sketches.py --im_name "ballerina" --layer_opt "4,11" --divs "0.45,0.9"
# ===================================================

# list of divs per layer
# background : layers = [2, 3, 4, 7, 8, 11] divs = [0.35, 0.45, 0.45, 0.45, 0.5, 0.9]
# foreground : layers = [2, 3, 4, 7, 8, 11] divs = [0.45, 0.45, 0.45, 0.4,  0.5, 0.9]


start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=str, default="4,11")
parser.add_argument("--divs", type=str, default="0.45,0.9")
parser.add_argument("--fg_bg_separation", type=int, default=1)
parser.add_argument("--num_strokes", type=int, default=64,
                    help="number of strokes used to generate the sketch, this defines the level of abstraction.")
args = parser.parse_args()
num_sketches = 1


## manual delete me
args.im_name = "ballerina"
args.layer_opt = "2,8,11"
args.divs = "0.35,0.5,0.85"

def run_generate_fidelity_levels(object_or_background, resize_obj,num_iter,layer_opt,gpu_id, process_id):
    exit_code = sp.run(["python", "scripts/generate_fidelity_levels.py",
                            "--im_name", args.im_name,
                            "--layer_opt", str(layer_opt),
                            "--object_or_background", object_or_background,
                            "--num_iter", str(num_iter),
                            "--resize_obj", resize_obj,
                            "--num_sketches", str(num_sketches),
                            "--num_strokes",str(args.num_strokes),
                            "--gpu_id",str(gpu_id),
                            "--process_id",str(process_id),
                            "--fg_bg_separation", str(args.fg_bg_separation)])
    
def run_ratio(simp_level, div, layer_opt, object_or_background, resize_obj,gpu_id, process_id):
    exit_code = sp.run(["python", "scripts/run_ratio.py",
                            "--im_name", args.im_name,
                            "--layer_opt", str(layer_opt),
                            "--object_or_background", object_or_background,
                            "--resize_obj",str(resize_obj),
                            "--min_div", str(div),
                            "--simp_level", str(simp_level),
                            "--num_sketches", str(num_sketches),
                            "--num_strokes",str(args.num_strokes),
                            "--gpu_id",str(gpu_id),
                            "--process_id",str(process_id),
                            "--fg_bg_separation", str(args.fg_bg_separation)])

if __name__ == "__main__":
        mp.set_start_method("spawn")
        ncpus = 10
        P1 = mp.Pool(ncpus)
        P2 = mp.Pool(ncpus)
        shift_gpu =2
        
        layers = [int(l) for l in args.layer_opt.split(",")]
        divs = [float(d) for d in args.divs.split(",")]
       
        # run bg fidelity multiprocessing
        for i,l in enumerate(layers):
                num_iter = 1501        
                if not ((args.fg_bg_separation and os.path.exists(f"./results_sketches/{args.im_name}/runs/background_l{str(l)}_{args.im_name}_mask/points_mlp.pt")) \
                        or ((not args.fg_bg_separation) and os.path.exists(f"./results_sketches/{args.im_name}/runs/l{str(l)}_{args.im_name}/points_mlp.pt"))):
                        P1.apply_async(run_generate_fidelity_levels, ("background", str(0), num_iter,l,i+shift_gpu,i))

        # run fg fidelity multiprocessing   
        if args.fg_bg_separation:             
                for i,l in enumerate(layers):
                        num_iter = 1000
                        if int(l) < 8: # converge fater for shallow layers
                                num_iter = 600
                        if not os.path.exists(f"./results_sketches/{args.im_name}/runs/object_l{str(l)}_{args.im_name}/points_mlp.pt"):
                                P1.apply_async(run_generate_fidelity_levels, ("object", str(1), num_iter, l, i+shift_gpu, i+3))

        P1.close()
        P1.join()  # start processes

        print("debug stop")
                       
        j = -1
        # run bg simplicity multiprocessing
        for i, (l, div) in enumerate(zip(layers,divs)):
                        try:
                                P2.apply_async(run_ratio,(j, div, l, "background", 0,i+shift_gpu,i))
                                #def run_ratio(simp_level, div, layer_opt, object_or_background, resize_obj,gpu_id, process_id):

                        except Exception as e:
                                print(f"An error occurred: {e}")

        # run fg simplicity multiprocessing
        if args.fg_bg_separation:
                for i, (l,div) in enumerate(zip(layers,divs)):
                                try:
                                        P2.apply_async(run_ratio,(j, div, l, "object",1, i+shift_gpu,i+3))
                                        #def run_ratio(simp_level, div, layer_opt, object_or_background, resize_obj,gpu_id, process_id):
                                except Exception as e:
                                        print(f"An error occurred: {e}")

        P2.close()
        P2.join()  # start processes


        sp.run(["python", "scripts/combine_matrix.py", 
                "--im_name", args.im_name,
                "--layers", str(args.layer_opt),
                "--fg_bg_separation", str(args.fg_bg_separation)])

        total_time = time.time() - start_time
        print(f"Total time for 9x3 matrix: [{total_time:.3f}] seconds")
        logging.info(f'python scripts/run_9x3_sketches_3gpu.py --im_name {args.im_name} --layer_opt {args.layer_opt} --divs {args.divs}'
                f' --fg_bg_separation {args.fg_bg_separation} --num_strokes {args.num_strokes}')
        logging.info(f"Total time for 9x3 matrix: [{total_time/60:.2f}] minutes")