import os
import sys
import shutil

import json
from glob import glob
import numpy as np
import xml.etree.ElementTree as ET


def datasetIRON2PSDR(src_dir,dst_dir,light_out,xml_file,xml_out,mesh_name,n_lights = -1,):
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
    
    if n_lights == 1:
        lightdir = [lightdir[0]]
    list_lightidx = np.random.choice(len(lightdir),size=len(list_imgfile))
    cam_dict = json.load(open(os.path.join(src_dir,lightdir[0],"train/cam_dict_norm.json"),"r"))
    breakpoint()
    
    # Record Light position and save to file
    files_light = glob(os.path.join(src_dir,"*/light.txt"))
    light_pos = []
    for light_file in files_light:
        lines = open(light_file,'r').readlines()
        for line in lines[1:]:
            light_pos.append([float(i) for i in line[:-2].split(' ')])
        # light_pos.append([float(i) for i in lines[1][:-2].split(' ')])
    
    # breakpoint()
    # Update medium info
    file_medium = os.path.join(src_dir,"medium.txt")
    if not os.path.exists(file_medium):
        raise FileNotFoundError()
    f = open(file_medium,'r')
    lines = [l.split('\n')[0] for l in f.readlines()]
    albedo,sigma_t,_,_ = lines
    isRGB = "\\" in albedo
 
    def replace_element(xml_prev,root,new_tag="float"):
        element_to_change = xml_med
        new_element = ET.Element(new_tag)
        new_element.text = element_to_change.text
        for attrib_name, attrib_value in element_to_change.attrib.items():
            new_element.set(attrib_name, attrib_value)
        for child in list(element_to_change):
            new_element.append(child)
        
        # Find the index of the old element in the parent's list of children
        index = list(root).index(element_to_change)
        root.remove(element_to_change)
        root.insert(index, new_element)
        
    list_xml_med = root.findall("bsdf")[0].findall("rgb")
    for xml_med in list_xml_med:
        if xml_med.get("name") == "albedo":
            xml_med.set("value",albedo)
            if not isRGB:
                replace_element(xml_med, root.findall("bsdf")[0])
        elif xml_med.get("name") == "sigma_t":
            xml_med.set("value",sigma_t)
            if not isRGB:
                replace_element(xml_med, root.findall("bsdf")[0])

    # Update obj info
    list_xml_shape = root.findall("shape")[0].findall("string")
    for xml_shape in list_xml_shape:
        if xml_shape.get("name") == "filename":
            xml_shape.set("value",f"../../smoothshape/{mesh_name}.obj")

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
    name = "cylinder5"
    name = "botijo4"
    name = "botijo5"
    name = "head1"
    name = "buddha1"
    name = "kettle1"
    # name="horse1"
    name = sys.argv[1]
    
    # Step 1. Copy scene file
    datasetIRON2PSDR(
        src_dir=f"./data/{sys.argv[2]}",
         dst_dir=f"./data_kiwi_soap/realdata/{name}",
         light_out=f"./examples/scenes/light/lights-{name}",
         xml_file="./examples/scenes/head_out.xml",
         xml_out=f"./examples/scenes/{name}_out.xml",
         mesh_name=f"{sys.argv[3]}",
         n_lights=1)
    
    # # Step 2. Copy Script file
    # script_pth = "./examples/python/scripts/learn_custom.sh"
    # dst_pth = script_pth.replace("custom",name)
    # shutil.copy(script_pth,dst_pth)
    
# if __name__ == "__main__":
#     src_dir = "data/duck/light00/test/image"
#     dst_dir = "data_kiwi_soap/realdata/duck"
#     xml_out = "examples/scenes/duck_out.xml"
#     datasetIRON2PSDR_single(src_dir,dst_dir,xml_out)