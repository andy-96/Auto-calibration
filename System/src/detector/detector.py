import torch, torchvision

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import os, cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys

def get_masked_imgs(img, masks=None, boxes=None, box_scale=-1):
    imtype = img.dtype
    height = img.shape[0]
    width = img.shape[1]
    # pixel masks
    pms = []
    if masks is not None:
        for mask in masks:
            pm = img.copy()
            pm[mask == 0] = 0
            pms.append(pm)
    # bounding box masks
    bbms = []
    if boxes is not None:
        for box in boxes:
#             x_bl, y_bl, x_tr, y_tr = box
            x_1, y_1, x_2, y_2 = box
            x_bl, y_bl, x_tr, y_tr = x_1, height-y_2, x_2, height-y_1
            # scale box
            if box_scale > 0:
                box_width, box_height = x_tr-x_bl, y_tr-y_bl
                width_delta, height_delta = (box_scale-1)*box_width/2, (box_scale-1)*box_height/2
                x_bl, y_bl, x_tr, y_tr = x_bl-width_delta, y_bl-height_delta, x_tr+width_delta, y_tr+height_delta
            # clip values
            [x_bl, x_tr] = np.clip([x_bl, x_tr], 0, width-1)
            [y_bl, y_tr] = np.clip([y_bl, y_tr], 0, height-1)
            # create bounding box
            bbm = np.zeros(img.shape, dtype=imtype)
            bbm[int(height-y_tr):int(height-y_bl), int(x_bl):int(x_tr), :] = img[int(height-y_tr):int(height-y_bl), int(x_bl):int(x_tr), :]
            bbms.append(bbm)
    return pms, bbms

class Detectron2:
    def __init__(self, model='mask-rcnn', model_yaml=None):
        if not isinstance(model_yaml, str):
            if isinstance(model, str) and (model.lower() == 'mask-rcnn'):
                model_yaml = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                print("[Detectron2] Loading model: MASK R-CNN (R-50+FPN+3x)")
            elif isinstance(model, str) and (model.lower() == 'faster-rcnn'):
                model_yaml = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
                print("[Detectron2] Loading model: FASTER R-CNN (R-50+FPN+3x)")
            else:
                print("[Detectron2] Invalid model choice!")
                exit(0)
        else:
            print(f"[Detectron2] Loading model: {model_yaml}")
        cfg = get_cfg()
        if torch.cuda.is_available():
            cfg.MODEL.DEVICE = 'cuda'
            print("[Detectron2] Use GPU")
        else:
            cfg.MODEL.DEVICE = 'cpu'
            print("[Detectron2] Use CPU")
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
        self.predictor = DefaultPredictor(cfg)
        print("[Detectron2] Model loaded!")
        self.img_root = None
        self.img_dict = dict() # {name: tag}
        self.model_yaml = model_yaml
        
    def load_images(self, img_root):
        self.img_root = img_root
        # read images
        if os.path.isdir(img_root):
            img_names = sorted(os.listdir(img_root))
            if '.DS_Store' in img_names:
                img_names.remove('.DS_Store')
            cnt = 0
            for img_name in tqdm(img_names, desc='loading images'):
                img_path = img_root + '/' + img_name
                if os.path.isfile(img_path):
                    cnt += 1
                    sep = img_name.find('.', -5)
                    self.img_dict[img_name[:sep]] = img_name[sep+1:]
            print(f"[Detectron2.load_images] {cnt} images are loaded.")
        else:
            print(f"[Detectron2.load_images] {img_root} is not a folder!")
            exit(0)
    def instance_segmentation(self, save_root, box_summary_dir=None, pixel_summary_dir=None, box_scale=-1, item_id=11):
        # check save root
        if not os.path.isdir(save_root):
            os.mkdir(save_root)
        if pixel_summary_dir is not None:
            try:
                f = open(f'{pixel_summary_dir}', 'w')
                f.close()
            except:
                print(f"[Detectron2.instance_segmentation] pixel_summary_dir '{pixel_summary_dir}' is invalid!")
                pixel_summary_dir = None
        if box_summary_dir is not None:
            try:
                f = open(f'{box_summary_dir}', 'w')
                f.close()
            except:
                print(f"[Detectron2.instance_segmentation] box_summary_dir '{box_summary_dir}' is invalid!")
                box_summary_dir = None
        # predict
        num = 0
        num_box = 0
        num_pixel = 0
        for img_name, tag in tqdm(self.img_dict.items(), desc='instance segmentation'):
            im = cv2.imread(f'{self.img_root}/{img_name}.{tag}') # BGR
            outputs = self.predictor(im)
            idx = np.intersect1d(np.where(outputs["instances"].to("cpu").pred_classes.numpy()==item_id), np.where(outputs["instances"].to("cpu").scores.numpy()>0.95))
            if len(idx) == 0: # no items detected
                tqdm.write(f"[Detectron2.instance_segmentation] {img_name}.{tag} not detected!")
                continue
            box = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[idx, :]
            mask = outputs["instances"].to("cpu").pred_masks.numpy()[idx, :, :]
            pms, bbms = get_masked_imgs(im[:, :, ::-1], list(mask), list(box), box_scale)
            if pixel_summary_dir is not None:
                for idx, pm in enumerate(pms):
                    # plt.imsave(f'{save_root}/{img_name}-{idx}pixel.{tag}', pm)
                    cv2.imwrite(f'{save_root}/{img_name}-{idx}pixel.{tag}', pm[:, :, ::-1])
                    with open(f'{pixel_summary_dir}', 'a') as f:
                        f.write(f'{img_name}-{idx}\n')
            if box_summary_dir is not None:
                for idx, bbm in enumerate(bbms):
                    # plt.imsave(f'{save_root}/{img_name}-{idx}box.{tag}', bbm)
                    cv2.imwrite(f'{save_root}/{img_name}-{idx}box.{tag}', bbm[:, :, ::-1])
                    with open(f'{box_summary_dir}', 'a') as f:
                        f.write(f'{img_name}-{idx}\n')
            # count
            num += 1
            num_box += len(bbms)
            num_pixel += len(pms)
        print(f"[Detectron2.instance_segmentation] {num} images | {num_box} bounding boxes | {num_pixel} pixel-wised masks")
    def object_detection(self, save_root, box_summary_dir=None, box_scale=-1, item_id=11):
        # check save root
        if not os.path.isdir(save_root):
            os.mkdir(save_root)
        if box_summary_dir is not None:
            try:
                f = open(f'{box_summary_dir}', 'w')
                f.close()
            except:
                print(f"[Detectron2.object_detection] box_summary_dir '{box_summary_dir}' is invalid!")
                box_summary_dir = None
        # predict
        num = 0
        num_box = 0
        for img_name, tag in tqdm(self.img_dict.items(), desc='object detection'):
            im = cv2.imread(f'{self.img_root}/{img_name}.{tag}') # BGR
            outputs = self.predictor(im)
            idx = np.intersect1d(np.where(outputs["instances"].to("cpu").pred_classes.numpy()==item_id), np.where(outputs["instances"].to("cpu").scores.numpy()>0.95))
            if len(idx) == 0: # no items detected
                tqdm.write(f"[Detectron2.object_detection] {img_name}.{tag} not detected!")
                continue
            box = outputs["instances"].to("cpu").pred_boxes.tensor.numpy()[idx, :]
            pms, bbms = get_masked_imgs(im[:, :, ::-1], list(), list(box), box_scale)
            if box_summary_dir is not None:
                for idx, bbm in enumerate(bbms):
                    # plt.imsave(f'{save_root}/{img_name}-{idx}box.{tag}', bbm)
                    cv2.imwrite(f'{save_root}/{img_name}-{idx}box.{tag}', bbm[:, :, ::-1])
                    with open(f'{box_summary_dir}', 'a') as f:
                        f.write(f'{img_name}-{idx}\n') 
            # count
            num += 1
            num_box += len(bbms)
        print(f"[Detectron2.object_detection] {num} images | {num_box} bounding boxes")

    def detect_stop_signs(self, img_root, save_root, box_summary_dir=None, pixel_summary_dir=None, box_scale=-1):
        # load images
        self.load_images(img_root)
        # detect
        if self.model_yaml.find('InstanceSegmentation') != -1:
            self.instance_segmentation(save_root, box_summary_dir, pixel_summary_dir, box_scale)
        elif self.model_yaml.find('Detection') != -1:
            self.object_detection(save_root, box_summary_dir, box_scale)
        else:
            raise NotImplementedError(self.model_yaml)
        
if __name__ == '__main__':
    img_root = sys.argv[1]
    save_root = sys.argv[2]
    box_summary_dir = sys.argv[3]
    pixel_summary_dir = None
    detector = Detectron2(model=sys.argv[4])
    box_scale = float(sys.argv[5])
    detector.detect_stop_signs(img_root, save_root, box_summary_dir, pixel_summary_dir, box_scale)
    
    
    
    
    
    
    
