import os
import shutil
import json
from glob import glob
import numpy as np
import xml.etree.ElementTree as ET


def datasetIRON2PSDR(src_dir,dst_dir,light_out,xml_file,xml_out):
    """
    Transform IRON dataset structure to PSDR-CUDA dataset
    """
    # NOTE: skipping camdict file as this has already been created by the "single" function
    # TODO: add test set?
    # et = ET.parse("examples/scenes/head_out.xml")
    et = ET.parse(xml_file)
    root = et.getroot()
    lightdir = [a for a in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir,a))]
    list_imgfile = sorted(os.listdir(os.path.join(src_dir,lightdir[0],"train/image")))
    list_lightidx = np.random.choice(len(lightdir)-1,size=len(list_imgfile))
    cam_dict = json.load(open(os.path.join(src_dir,lightdir[0],"train/cam_dict_norm.json"),"r"))
    
    files_light = glob(os.path.join(src_dir,"*/light.txt"))
    light_pos = []
    for light_file in files_light:
        lines = open(light_file,'r').readlines()
        light_pos.append([float(i) for i in lines[1][:-2].split(' ')])
    
    # Record Light position and save to file
    # Save medium info

    os.makedirs(dst_dir,exist_ok = True)
    os.makedirs(os.path.join(dst_dir,"exr_ref"),exist_ok = True)

    # Move images
    # img_files = sorted(os.listdir(src_dir))
    list_lightpos = []
    for idx,img_file in enumerate(list_imgfile):
        dir_light = lightdir[list_lightidx[idx]]
        src_file = os.path.join(src_dir,dir_light,"train/image",img_file)
        print(src_file)
        # cam = cam_dict[img_file.replace("exr","png")]
        print(img_file+"->",idx)
        dst_file = os.path.join(dst_dir,"exr_ref",f"{idx}.exr")
        print(dst_file)
        shutil.copy(src_file,dst_file)
        list_lightpos.append(light_pos[list_lightidx[idx]])
        
        cam = cam_dict[img_file.replace("exr","png")]
        K = np.array(cam["K"]).reshape(4,4)[:3,:3]
        W2C = np.array(cam["W2C"]).reshape(4,4)
        C2W = np.linalg.inv(W2C)

        # TODO: think this through
        up = -C2W[:3,1]
        up = up / np.linalg.norm(up)
        forward = C2W[:3,2]
        origin = C2W[:3,3]
        target = forward + origin  #[0,0,0] #origin + forward # should this be direction?
        xml_lookat = root.findall("sensor")[idx].find("transform").find("lookat")
        xml_lookat.set("origin", f"{origin[0]}, {origin[1]}, {origin[2]}")
        xml_lookat.set("target", f"{target[0]}, {target[1]}, {target[2]}")
        xml_lookat.set("up",f"{up[0]}, {up[1]}, {up[2]}")
        
        img_size = 512
        focal = K[0][0]
        fov = 2*np.tanh(img_size / 2 / focal)
        fov = np.rad2deg(fov)
        root.findall("sensor")[idx].find("float").set("value",f"{fov}")
    
    et.write(xml_out) 
    # Set light
    np.save(light_out,np.array(list_lightpos))


def datasetIRON2PSDR_single(src_dir,dst_dir,xml_out):
    # Set camera pose
    cam_dict = json.load(open(os.path.join(src_dir,"../cam_dict_norm.json"),"r"))
    et = ET.parse("examples/scenes/duck.xml")
    root = et.getroot()
    os.makedirs(dst_dir,exist_ok = True)
    os.makedirs(os.path.join(dst_dir,"exr_ref"),exist_ok = True)
    
    # Move images
    img_files = sorted(os.listdir(src_dir))
    for idx,img_file in enumerate(img_files):
        cam = cam_dict[img_file.replace("exr","png")]
        K = np.array(cam["K"]).reshape(4,4)[:3,:3]
        W2C = np.array(cam["W2C"]).reshape(4,4)
        C2W = np.linalg.inv(W2C)

        # TODO: think this through
        # Why is the up direction flipped..?
        up = -C2W[:3,1]
        # up = C2W[:3,1]
        up = up / np.linalg.norm(up)
        # forward = -C2W[:3,2]
        forward = C2W[:3,2]
        origin = C2W[:3,3]
        target = forward + origin  #[0,0,0] #origin + forward # should this be direction?
        xml_lookat = root.findall("sensor")[idx].find("transform").find("lookat")
        xml_lookat.set("origin", f"{origin[0]}, {origin[1]}, {origin[2]}")
        xml_lookat.set("target", f"{target[0]}, {target[1]}, {target[2]}")
        xml_lookat.set("up",f"{up[0]}, {up[1]}, {up[2]}")
        
        img_size = 512
        focal = K[0][0]
        fov = 2*np.tanh(img_size / 2 / focal)
        fov = np.rad2deg(fov)
        root.findall("sensor")[idx].find("float").set("value",f"{fov}")

        print(img_file+"->",idx)
        dst_file = os.path.join(dst_dir,"exr_ref",f"{idx}.exr")
        print(dst_file)
        shutil.copy(os.path.join(src_dir,img_file),dst_file)
    
    et.write(xml_out) 

    

if __name__ == "__main__":
    datasetIRON2PSDR(src_dir="./data/head/head",
                     dst_dir="./data_kiwi_soap/realdata/head",
                     light_out="./examples/scenes/light/lights-head",
                     xml_file="./examples/scenes/head_out.xml",
                     xml_out="./examples/scenes/head_out_2.xml")
    
# if __name__ == "__main__":
#     src_dir = "data/duck/light00/test/image"
#     dst_dir = "data_kiwi_soap/realdata/duck"
#     xml_out = "examples/scenes/duck_out.xml"
#     datasetIRON2PSDR_single(src_dir,dst_dir,xml_out)