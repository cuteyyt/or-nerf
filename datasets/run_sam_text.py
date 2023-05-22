"""
Text prompt for sam
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Grounding DINO
sys.path.append(os.path.join(os.getcwd(), "prior/Grounded-Segment-Anything"))
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.colmap.read_write_model import read_model


def load_image(image_path):
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
    model.eval()
    return model


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def draw_masks_on_img(img, masks, out_path):
    masks = masks.astype(np.int32)
    img[masks >= 1] = 255

    cv2.imwrite(out_path, img)


def save_masks(masks, out_path):
    masks[masks >= 1] = 255
    masks[masks < 1] = 0
    masks_img = masks.astype(np.uint8)
    cv2.imwrite(out_path, masks_img)


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

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

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


def parse():
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)

    parser.add_argument("--in_dir", type=str, )
    parser.add_argument("--out_dir", type=str, )
    parser.add_argument("--dataset", type=str, help="dataset name")
    parser.add_argument("--scene", type=str, help="scene name")
    parser.add_argument("--json_path", type=str, help="path to text prompt json")

    parser.add_argument("--config", type=str,
                        default='prior/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                        help="path to config file")
    parser.add_argument("--grounded_checkpoint", type=str,
                        default='ckpts/grounded_sam/groundingdino_swint_ogc.pth',
                        help="path to checkpoint file")
    parser.add_argument("--sam_checkpoint", type=str,
                        default='ckpts/sam/sam_vit_h_4b8939.pth',
                        help="path to checkpoint file"
                        )

    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--hybrid", action='store_true', help="combine text prompt with points")

    parser.add_argument("--is_test", action="store_true")

    args = parser.parse_args()

    return args


# noinspection PyPep8Naming
def main():
    args = parse()

    in_dir, out_dir = args.in_dir, args.out_dir
    dataset, scene = args.dataset, args.scene
    json_path = args.json_path

    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    sam_checkpoint = args.sam_checkpoint

    device = args.device
    hybrid = args.hybrid
    is_test = args.is_test

    in_dir = os.path.join(in_dir, f'{dataset}_sparse', scene)
    out_dir = os.path.join(out_dir, f'{dataset}_sam_text', scene)

    os.makedirs(out_dir, exist_ok=True)

    # open json
    with open(json_path, 'r') as file:
        text_params = json.load(file)
        text_prompt_list = text_params[dataset][scene]["text"]
        box_threshold = text_params[dataset][scene]["box_threshold"]
        text_threshold = text_params[dataset][scene]["text_threshold"]
        factor = text_params[dataset][scene]["factor"]

    # load model
    model = load_model(config_file, grounded_checkpoint, device=device)

    cam_dir = os.path.join(in_dir, 'sparse/0')
    _, images, _ = read_model(path=cam_dir, ext='.bin')

    print('Predict all views\' masks by SAM according to text prompt')
    with tqdm(total=len(images)) as t_bar:
        for image_id, image in images.items():
            image_filename = Path(image.name)
            img_path = os.path.join(in_dir, 'images_{}'.format(factor), image_filename)

            img = cv2.imread(img_path)
            img_pil_tensor, img_tensor = load_image(img_path)
            mask = np.zeros((img.shape[:2])).astype(np.int32)

            for text_prompt in text_prompt_list:

                # run grounding dino model
                boxes_filt, pred_phrases = get_grounding_output(
                    model, img_tensor, text_prompt, box_threshold, text_threshold, device=device)

                # initialize SAM
                predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
                predictor.set_image(img, image_format='BGR')

                H, W = img.shape[:2]
                for i in range(boxes_filt.size(0)):
                    boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
                    boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
                    boxes_filt[i][2:] += boxes_filt[i][:2]

                boxes_filt = boxes_filt.cpu()
                transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, img.shape[:2]).to(device)

                if len(transformed_boxes) != 0:
                    logits = None
                    for i in range(5):
                        masks, _, logits = predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes.to(device),
                            mask_input=logits,
                            multimask_output=False,
                            return_logits=True
                        )
                    # print(masks.shape)

                    os.makedirs(os.path.join(out_dir, 'masks'), exist_ok=True)
                    os.makedirs(os.path.join(out_dir, 'imgs_with_masks'), exist_ok=True)
                    os.makedirs(os.path.join(out_dir, 'imgs_with_boxes'), exist_ok=True)

                    mask_path = os.path.join(out_dir, 'masks', image_filename)
                    img_with_mask_path = os.path.join(out_dir, 'imgs_with_masks', image_filename)
                    img_with_box_path = os.path.join(out_dir, 'imgs_with_boxes', image_filename)

                    masks = masks.cpu().numpy()
                    for i in range(masks.shape[0]):
                        mask += masks[i][0].astype(np.int32)

                    save_masks(mask.copy(), mask_path)
                    draw_masks_on_img(img.copy(), mask.copy(), img_with_mask_path)

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img)
                    for box, label in zip(boxes_filt, pred_phrases):
                        # print(box, label)
                        show_box(box.numpy(), plt.gca(), label)
                        plt.axis('off')
                        plt.savefig(
                            img_with_box_path,
                            bbox_inches="tight", dpi=300, pad_inches=0.0
                        )

            t_bar.update(1)
            if hybrid:
                print("-----------------Generate first mask only, check points prompt-----------------")
                exit(0)


if __name__ == "__main__":
    main()
