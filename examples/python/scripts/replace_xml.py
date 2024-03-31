# import xml
import sys
import argparse
import xml.etree.ElementTree as ET
 
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
    
if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma_t',           type=str)
    parser.add_argument('--albedo',          type=str)
    parser.add_argument('--in_xml',           type=str)
    parser.add_argument('--out_xml',          type=str)
    parser.add_argument('--is_baseline', default=False,action="store_true")

    args = parser.parse_args()

    et = ET.parse(args.in_xml)
    root = et.getroot()
    list_xml_med = root.findall("bsdf")[0].findall("rgb")
    for xml_med in list_xml_med:
        if xml_med.get("name") == "albedo":
            xml_med.set("value",args.albedo)
        elif xml_med.get("name") == "sigma_t":
            xml_med.set("value",args.sigma_t)

    list_xml_med = root.findall("bsdf")[0].findall("float")
    for xml_med in list_xml_med:
        if xml_med.get("name") == "albedo":
            xml_med.set("value",args.albedo)
        elif xml_med.get("name") == "sigma_t":
            xml_med.set("value",args.sigma_t)
    if args.is_baseline:
        xml_med = root.findall("bsdf")[0]
        xml_med.set("type","hetersub")

    et.write(args.out_xml)


        