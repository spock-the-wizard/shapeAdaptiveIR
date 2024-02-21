import os
import shutil
import json
from glob import glob
import numpy as np
import xml.etree.ElementTree as ET

if __name__ == "__main__":
    src_dir = "data/duck/light00/test/image"
    dst_dir = "data_kiwi_soap/realdata/duck"
    xml_out = "examples/scenes/duck_out.xml"
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
        # shutil.copy(os.path.join(src_dir,img_file),dst_file)
        
    et.write(xml_out)
    # Set light
