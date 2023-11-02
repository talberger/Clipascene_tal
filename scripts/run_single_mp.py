import subprocess as sp
import argparse
import time
import os
import multiprocessing as mp


start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--im_name", type=str, default="")
parser.add_argument("--layer_opt", type=int, default=4)
# parser.add_argument("--resize_obj", type=int, default=0)
# parser.add_argument("--num_iter", type=int, default=1501)
args = parser.parse_args()

#manual for debug#
# args.im_name = "bull"
# args.layer_opt = 4
## end manual ##




def run(object_or_background, resize_obj,num_iter):
    exit_code = sp.run(["python", "scripts/generate_fidelity_levels.py",
                            "--im_name", args.im_name,
                            "--layer_opt", str(args.layer_opt),
                            "--object_or_background", object_or_background,
                            "--num_iter", str(num_iter),
                            "--resize_obj", resize_obj])


if __name__ == "__main__":
        mp.set_start_method("spawn")
        ncpus = 10
        P = mp.Pool(ncpus)


        num_iter = 1501        
        if not os.path.exists(f"./results_sketches/{args.im_name}/runs/background_l{str(args.layer_opt)}_{args.im_name}_mask/points_mlp.pt"):
                P.apply_async(run, ("background", str(0), num_iter))

        num_iter = 1000
        if args.layer_opt < 8: # converge fater for shallow layers
                num_iter = 600
        if not os.path.exists(f"./results_sketches/{args.im_name}/runs/object_l{str(args.layer_opt)}_{args.im_name}/points_mlp.pt"):
                P.apply_async(run, ("object", str(1), num_iter))

        P.close()
        P.join()  # start processes

        # python scripts/combine_matrix.py --im_name <im_name>
        sp.run(["python", "scripts/combine_matrix.py", 
                "--im_name", args.im_name,
                "--layers", str(args.layer_opt),
                "--rows", "1",
                "--is_single", "1"])

        total_time = time.time() - start_time
        print(f"Time run single sketch [{total_time:.3f}] seconds")