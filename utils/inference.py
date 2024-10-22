import os
import cv2
import torch
import time
from collections import OrderedDict
from PIL import Image
import torchvision.transforms as transforms
def inference_speed(model):
    inp_tensor = torch.ones(1,3,256,256)
    test_results = OrderedDict()
    test_results["runtime"] = []
    test_results["cputime"] = []
    with torch.no_grad():
        for _ in range(10):
            o = model(inp_tensor)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(50):
            start_time = time.time()
            start.record()
            _ = model(inp_tensor)
            end.record()
            end_time = time.time()
            torch.cuda.synchronize()
            spend_time = start.elapsed_time(end)
            cpu_time = end_time - start_time
            test_results["runtime"].append(spend_time)
            test_results["cputime"].append(cpu_time)
        ave_runtime = sum(test_results["runtime"]) / len(test_results["runtime"])
        cpu_avg = sum(test_results["cputime"]) / len(test_results["cputime"])
        print("Average Time:{:.4f} ms, cpu Time:{:.4f} ms".format(ave_runtime, cpu_avg*1000))

def get_gt_label(txt_path = r"D:\datasets\\imagenet_val\\val.txt"):
    # 打开txt文件，读取每一行内容
    hashtable = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # print(line.strip(" "))
            line = line.split(" ")
            name = line[0]
            label = line[1]
            hashtable[name] = int(label)
    return hashtable

def inference_imagenet(model, 
                       root_path = r"D:\datasets\\imagenet_val\\ILSVRC2012_img_val", 
                       cnt=100):
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
])
    img_list = os.listdir(root_path)
    hashtable = get_gt_label()
    acc = 0
    for i,img in enumerate(img_list):
        label = hashtable[img]
        img_path = os.path.join(root_path, img)
        image = Image.open(img_path).convert("RGB")
        with torch.no_grad():
            img_tensor = trans(image).unsqueeze(0)
            out = model(img_tensor)
            probs = torch.softmax(out, dim=1)
            value, index = probs.max(dim=1)
            if index == label:
                acc += 1
            # print(f"img: {img}, label: {label}, pred: {index}")
        if i >= cnt:
            break
    print(f"acc: {acc/cnt}")