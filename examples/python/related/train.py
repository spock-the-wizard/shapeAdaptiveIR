import matplotlib.pyplot as plt

import drjit as dr
import mitsuba as mi

import os
import time
import random
from datetime import datetime
import argparse
import sys

import xml.etree.ElementTree as ET

import numpy as np
import cv2
from tqdm import tqdm
import wandb

# os.chdir("../")
# root_code="/home/spock-the-wizard/slurm/sss-relighting/InverseTranslucent/examples/python"
# os.chdir(root_code)
# sys.path.append(f"{root_code}")
from constants import params_gt
            # import mitsuba.llvm_ad_rgb.Color3f as Color3f

def scene2mesh(scene):
    if 'croissant' in scene:
        return "../../examples/smoothshape/final/croissant.obj" 
    elif 'duck' in scene:
        return "../../examples/smoothshape/duck_v2.obj" 
    elif 'gargoyle' in scene:
        return "../../examples/smoothshape/final/gargoyle.obj" 
    elif 'head' in scene:
        return "../../examples/smoothshape/final/head_.obj" 
    elif 'maneki' in scene:
        return "../../examples/smoothshape/final/maneki_.obj" 
    elif 'botijo' in scene:
        return "../../examples/smoothshape/final/botijo2_.obj" 
    elif 'torus' in scene:
        return "../../examples/smoothshape/final/torus.obj" 
    elif 'dragon' in scene:
        return "../../examples/smoothshape/final/dragon.obj" 
    elif 'kettle' in scene:
        return "../../examples/smoothshape/final/kettle_.obj" 

def replace_element(xml_pth,out_pth,str_nodes,value,):
    """ Replace nested element with given value """
    et = ET.parse(xml_pth)
    root = et.getroot()

    node = root
    for str_node in str_nodes:
        node = node.findall(str_node)[0]

    node.set("value",value)

    et.write(out_pth)
    

mi.set_variant('llvm_ad_rgb')
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

parser = argparse.ArgumentParser()
parser.add_argument('--scene',              type=str,      default='sphere1')
parser.add_argument('--exp_name',           type=str,      default=None)
parser.add_argument('--spp',                type=int,      default=8)
parser.add_argument('--ref_spp',                type=int,      default=128)

parser.add_argument('--sweep_num',                type=int,      default=0)

parser.add_argument('--n_iters',            type=int,      default=300)
parser.add_argument('--n_dump',            type=int,      default=100)
parser.add_argument('--n_log',            type=int,      default=10)
parser.add_argument('--lr',            type=float,      default=0.05)

parser.add_argument('--debug', action="store_true", default=False)
parser.add_argument('--render', action="store_true", default=False)
parser.add_argument('--onlySig', action="store_true", default=False)


def wandb_log(*wandb_args,**kwargs):
    if not args.debug:
        wandb.log(*wandb_args,**kwargs)
def rmse(predictions, targets):
    # Computes error map for each pixel
    return np.sqrt(((predictions - targets) ** 2).mean(axis=-1))

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.debug:
        # run = wandb.init(
        #         project="pg2024_rebuttal_prb",
        #         config=args
        # )
        run = wandb.init(
            project="pg2024_rebuttal_prb",
        )
        args.exp_name = args.exp_name+f'_{run.name}'
        wandb.config.update(args)
        for key,val in wandb.config.items():
            if hasattr(args,key):
                setattr(args,key,val)
        args.scene = wandb.config['scene']
        print(wandb.config['scene'])
        print(args.scene)


    # Load reference images
    ref_dir = f"/home/spock-the-wizard/slurm/sss-relighting/InverseTranslucent/data_kiwi_soap/realdata/{args.scene}/exr_ref"
    if args.scene == "duck":
        scene_file = f"./related/scene/duck_out.xml"
    else:
        scene_file = f"./related/scene/croissant1_out.xml"
        # Replace obj of scene_file
        out_file = scene_file.replace("croissant1",args.scene)
        replace_element(scene_file,out_file,["shape","string"],scene2mesh(args.scene))
        scene_file = out_file

    list_refs = []
    for i in range(len(os.listdir(ref_dir))):
        img_pth = f"{ref_dir}/{i}.exr"
        img = cv2.imread(img_pth,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        list_refs.append(img)
    
    scene = mi.load_file(scene_file)
    params = mi.traverse(scene)

    light_pth = f"../scenes/light/lights-{args.scene}.npy"
    try:
        lights = np.load(light_pth)
    except:
        if args.scene == "duck":
            lights = np.array([[1.192, -1.3364, 0.889]]).repeat(25,axis=0)

    for key,value in params.items():
        if key.startswith('sensor') and key.endswith('film.size'):
            params[key] = (512,512)
    sensors = scene.sensors()


    # Prepare for training
    sensor_count = len(os.listdir(ref_dir))
    iteration_count = args.n_iters
    spp = args.spp
    ref_spp = args.ref_spp

    gt_alb = np.array(params_gt[args.scene]['albedo'])
    gt_sig = np.array(params_gt[args.scene]['sigmat'])

    statsroot = './related/results'
    curtime = datetime.now().strftime("%y%m%d_%H%M%S")
    statsdir = f"{statsroot}/{args.scene}_{curtime}"
    statsdir = f"{statsroot}/{args.scene}/{args.exp_name}"
    os.makedirs(statsdir,exist_ok=True)
            
    # Training
    key_sig = 'OBJMesh.interior_medium.sigma_t.value.value'
    key_alb = 'OBJMesh.interior_medium.albedo.value.value'
    key_scale = 'OBJMesh.interior_medium.scale'

    # Initialize medium parameters
    if args.sweep_num != -1:
        import pickle
        with open("../random_init_240717_050427.pickle",'rb') as f:
            random_inits = pickle.load(f)
            
            init_albedo = random_inits['init_albedo'][args.sweep_num] 
            init_sigmat = random_inits['init_sigmat'][args.sweep_num] 

            params[key_sig] = init_sigmat / params[key_scale]
            params[key_alb] = init_albedo
    else:
        params[key_sig] = np.array([0.5,0.5,0.5])
        params[key_alb] = np.array([0.5,0.5,0.5])

        # # maneki / PRB (64 spp)
        # params[key_sig] = np.array([0.62, 0.50, 0.50])
        # params[key_alb] = np.array([0.89, 0.88, 0.88])

        # # maneki / PRB (32 spp)
        # params[key_sig] = np.array([0.31, 0.99, 0.99])
        # params[key_alb] = np.array([0.89, 0.88, 0.88])

        # maneki / PRB (32 spp)
        params[key_sig] = np.array([0.50, 0.53, 0.53])
        params[key_alb] = np.array([0.85, 0.88, 0.88])


    # # Debugging forward rendering
    # opt = mi.ad.Adam(lr=0.001)
    # params[key_alb] = gt_alb
    # params[key_sig] = gt_sig /  params[key_scale]

    opt = mi.ad.Adam(lr=args.lr)
    opt[key_sig] = params[key_sig]
    if not args.onlySig:
        opt[key_alb] = params[key_alb]
    params.update(opt)

    list_times = []

    sensor_list =[11,]
    it = 1
    if args.render:
        print(f"Render Images")
        for sensor_idx in sensor_list:
            params['point-light.position'] = lights[sensor_idx]
            img_pth = f"{statsdir}/iter{it}_{sensor_idx}_out.exr"
            img = mi.render(scene, params, sensor=sensor_idx, spp=ref_spp, seed=it)
            mi.util.write_bitmap(img_pth,img)

            gt_pth = f"{statsdir}/iter{it}_{sensor_idx}_gt.exr"
            gt = list_refs[sensor_idx]
            mi.util.write_bitmap(gt_pth,gt)
            print(f"Wrote image {sensor_idx}")

    else:
        for it in tqdm(range(iteration_count)):
            total_loss = 0.0

            
            if ((it+1)%args.n_dump) == 0:
                list_rmse = []
                for sensor_idx in [0,1,4,10]:
                    params['point-light.position'] = lights[sensor_idx]
                    img_pth = f"{statsdir}/iter{it}_{sensor_idx}_out.exr"
                    img = mi.render(scene, params, sensor=sensor_idx, spp=ref_spp, seed=it)
                    mi.util.write_bitmap(img_pth,img)

                    gt_pth = f"{statsdir}/iter{it}_{sensor_idx}_gt.exr"
                    gt = list_refs[sensor_idx]
                    mi.util.write_bitmap(gt_pth,gt)
                    list_rmse.append(rmse(img.numpy(),gt).mean())

                param_sig = params[key_sig].numpy() * params[key_scale]
                wandb_log({
                    # "test/loss": total_loss,
                    "test/epoch": it,
                    "test/rmse_alb" : rmse(params[key_alb].numpy(),gt_alb),
                    "test/rmse_sig" : rmse(param_sig,gt_sig),
                    "test/rmse_img" : np.array(list_rmse).mean()
                    },step=it)
                continue
            
            time_start = time.time()

            sensor_idx = random.randint(0,sensor_count-1) 
            # print(sensor_idx)
            params['point-light.position'] = lights[sensor_idx]
            img = mi.render(scene, params, sensor=sensor_idx, spp=spp, seed=it)
            # img = mi.render(scene, params, sensor=0, spp=spp, seed=it)
            # tmp = img.numpy() * 255
            # cv2.imwrite('test3.png',tmp)
            # breakpoint()

            loss = dr.mean(dr.sqr(img - list_refs[sensor_idx]))
            dr.backward(loss)
            opt.step()

            # Clamp the optimized density values. Since we used the `scale` parameter
            # when instantiating the volume, we are in fact optimizing extinction
            # in a range from [1e-6 * scale, scale].
            opt[key_sig] = dr.clamp(opt[key_sig], 1e-6, 1.0)
            if not args.onlySig:
                opt[key_alb] = dr.clamp(opt[key_alb], 1e-6, 1.0)
            # else:
            #     opt[key_sig] = Color3f(opt[key_sig].numpy().mean())
            print(opt[key_sig])

            # Propagate changes to the scene
            params.update(opt)
            time_end = time.time()
            elapsed_time = time_end - time_start
            list_times.append(elapsed_time)
            avg_time = sum(list_times) / len(list_times)

            total_loss += loss[0]
            param_sig = params[key_sig].numpy() * params[key_scale]
            wandb_log({
                "train/loss": total_loss,
                "train/epoch": it,
                "train/rmse_alb" : rmse(params[key_alb].numpy(),gt_alb),
                "train/rmse_sig" : rmse(param_sig,gt_sig),
                "train/time": avg_time,
                "param/sigmaT_r": params[key_sig][0].numpy(),
                "param/sigmaT_g": params[key_sig][1].numpy(),
                "param/sigmaT_b": params[key_sig][2].numpy(),
                "param/albedo_r": params[key_alb][0].numpy(),
                "param/albedo_g": params[key_alb][1].numpy(),
                "param/albedo_b": params[key_alb][2].numpy(),
                },step=it)

            # if it % args.n_log == 0: 

            tqdm.write(f"Iteration {it:02d}: error={total_loss:6f}", end='\r')