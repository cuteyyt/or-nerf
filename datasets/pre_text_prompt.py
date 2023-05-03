import argparse
import os
import copy
import sys
import json

import numpy as np
import json
import torch
from pathlib import Path
from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
sys.path.append(os.path.join(os.getcwd(), "prior/Grounded-Segment-Anything"))
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.colmap.read_write_model import read_model

def save_masks(masks, out_path):
    masks_img = (masks * 255).astype(np.uint8)
    cv2.imwrite(out_path, masks_img)


def draw_masks_on_img(img, masks, out_path):
    masks = masks.astype(np.int32)
    img[masks >= 1] = 255

    cv2.imwrite(out_path, img)

def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1] # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    # with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
    #     json.dump(json_data, f)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument(
        "--grounded_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--input_image", type=str, help="path to image file")
    parser.add_argument("--text_prompt", type=str, help="text prompt")
    parser.add_argument("--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--scene", type=str, required=True, help="scene name")
    #parser.add_argument("--box_threshold", type=float, default=0.3, help="box threshold")
    #parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--text_prompt_json", type=str, help="path to text prompt json")

    parser.add_argument("--device", type=str, default="cpu", help="running on cpu only!, default=False")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint
    image_path = args.input_image
    #input_dir = args.input_dir
    text_prompt = args.text_prompt
    #output_dir = args.output_dir
    dataset = args.dataset
    scene = args.scene
    #box_threshold = args.box_threshold
    #text_threshold = args.text_threshold
    text_prompt_json_path = args.text_prompt_json
    device = args.device


    input_dir = "data/{}_sparse/{}".format(dataset, scene)
    output_dir = "data/{}_sam/{}".format(dataset, scene)
    # make dir
    os.makedirs(output_dir, exist_ok=True)
    # open json
    with open(args.text_prompt_json, 'r') as file:
        text_params = json.load(file)
        text_prompt_list = text_params[dataset][scene]["text"]
        box_threshold = text_params[dataset][scene]["box_threshold"]
        text_threshold = text_params[dataset][scene]["text_threshold"]
        factor = text_params[dataset][scene]["factor"]

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    cam_dir = os.path.join(input_dir, 'sparse/0')
    _, images, _= read_model(path=cam_dir, ext='.bin')


    with tqdm(total=len(images) - 1) as t_bar:
        for image_id, image in images.items():
            image_filestem = Path(image.name)
            print(image_filestem)
            # if str(image_filestem) != "20220811_113229.jpg":
            #     t_bar.update(1)
            #     continue
            image_path = os.path.join(input_dir, 'images_{}'.format(factor), image_filestem)
            if dataset == 'spinnerf_dataset':
                image_path = image_path[:-3] + 'png'
            if not os.path.exists(image_path):
                t_bar.update(1)
                continue
            image = cv2.imread(image_path)
            mask = np.zeros((image.shape[0], image.shape[1])).astype(np.int32)
            for text_prompt in text_prompt_list:
                # load image
                #print(text_prompt)
                image_pil, image = load_image(image_path)

                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    model, image, text_prompt, box_threshold, text_threshold, device=device
                )

                # initialize SAM
                predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
                image = cv2.imread(image_path)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                predictor.set_image(image, image_format='BGR')

                H, W = image.shape[:2]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                boxes_filt = boxes_filt.cpu()
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
                print(transformed_boxes)
                

                if len(transformed_boxes) != 0 :
                    masks, _, _ = predictor.predict_torch(
                        point_coords = None,
                        point_labels = None,
                        boxes = transformed_boxes.to(device),
                        multimask_output = False,
                    )
                    print(masks.shape)

                    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, 'imgs_with_masks'), exist_ok=True)
                    os.makedirs(os.path.join(output_dir, 'imgs_with_boxes'), exist_ok=True)
                    mask_dir = os.path.join(output_dir, 'masks', image_filestem)
                    mask_img_dir = os.path.join(output_dir, 'imgs_with_masks', image_filestem)
                    box_img_dir = os.path.join(output_dir, 'imgs_with_boxes', image_filestem)
                    masks = masks.cpu().numpy()

                    mask += masks[0][0].astype(np.int32) 
                    save_masks(mask.copy(), mask_dir)
                    draw_masks_on_img(image.copy(), mask.copy(), mask_img_dir)
                    
                    plt.figure(figsize=(10,10))
                    plt.imshow(image)
                    for box, label in zip(boxes_filt, pred_phrases):
                        print(box, label)
                        show_box(box.numpy(), plt.gca(), label)
                        plt.axis('off')
                        plt.savefig(
                            box_img_dir, 
                            bbox_inches="tight", dpi=300, pad_inches=0.0
                        )

            t_bar.update(1)

        # draw output image



        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        # for box, label in zip(boxes_filt, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)

        # plt.axis('off')
        # plt.savefig(
        #     os.path.join(output_dir, "imgs_with_mask", "grounded_sam_output.jpg"), 
        #     bbox_inches="tight", dpi=300, pad_inches=0.0
        # )

        # save_mask_data(output_dir, masks, boxes_filt, pred_phrases)

