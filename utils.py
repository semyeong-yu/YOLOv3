import argparse
import random
import torch
import numpy as np
import os
import sys

def arg_parse():
    parser = argparse.ArgumentParser(description="YOLOv3")

    # mode
    parser.add_argument("--mode", type=str, default="train", required=True) # 'train' or 'test'
    
    # random seed
    parser.add_argument("--random_seed", type=int, default=0)

    # optimizer
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--lr", type=float, default="0.01")
    parser.add_argument("--beta1", type=float, default="0.9")
    parser.add_argument("--beta2", type=float, default="0.99")

    # lr scheduler
    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau")

    # checkpoint dir
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint")
    # checkpoint to resume training
    parser.add_argument("--checkpoint_train", type=str, default="./checkpoint/0825/latest.pt")
    # checkpoint for to start test
    parser.add_argument("--checkpoint_best", type=str, default="./checkpoint/0825/best.pt")

    # setting
    parser.add_argument("--date", type=str, default="MMDD")

    # dataset
    parser.add_argument("--crop_size", nargs=2, type=int, default=[416, 416]) # crop이 아니라 resize
    parser.add_argument("--n_class", type=int, default=80) # 80 classes for COCO dataset
    parser.add_argument("--data_dir", type=str, default='./dataset')
    parser.add_argument("--img_dir_train", type=str, default='images/train2017_small') # train2017small은 train2017의 사진들 중 일부를 발췌함
    parser.add_argument("--img_dir_val", type=str, default='images/val2017_small')
    parser.add_argument("--img_dir_test", type=str, default='images/test2017')
    parser.add_argument("--gt_json_train", type=str, default='annotations/instances_train2017.json')
    parser.add_argument("--gt_json_val", type=str, default='annotations/instances_val2017.json')

    # train/test
    parser.add_argument("--no_resume", action='store_true') # no_resume = True if specified in argument
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_epoch", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--accumulate_iter", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=5.)
    parser.add_argument("--bbox_weight", type=float, default=0.1)
    parser.add_argument("--object_weight", type=float, default=0.1)
    parser.add_argument("--class_weight", type=float, default=0.1)
    parser.add_argument("--big_anchor_size", nargs=6, type=int, default=[116, 90, 156, 198, 373, 326], help='A list of 3 anchor sizes for 13x13 small feature map')
    parser.add_argument("--middle_anchor_size", nargs=6, type=int, default=[30, 61, 62, 45, 59, 119], help='A list of 3 anchor sizes for 26x26 middle feature map')
    parser.add_argument("--small_anchor_size", nargs=6, type=int, default=[10, 13, 16, 30, 33, 23], help='A list of 3 anchor sizes for 52x52 big feature map')
    parser.add_argument("--iou_thres", type=float, default=0.5)
    parser.add_argument("--NMS_thres", type=float, default=0.3)
    parser.add_argument("--acc_obj_thres", type=float, default=0.3)
    parser.add_argument("--acc_iou_thres", type=float, default=0.5)

    return parser.parse_args()

def fix_seed(random_seed):
    torch.manual_seed(random_seed) # for CPU
    torch.cuda.manual_seed(random_seed) # for GPU
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True # cuDNN uses deterministic algorithm: 동일한 입력과 초기 조건에서 항상 같은 결과를 보장
    torch.backends.cudnn.benchmark = False # input data size가 변하는 경우 cuDNN은 다양한 algorithm을 benchmark하여 가장 빠른 것을 선택하는데, 이는 재현성에 부정적인 영향을 줌. 입력 크기가 고정되어 있거나 재현성이 중요한 경우 cuDNN의 benchmark 기능을 끔
    np.random.seed(random_seed)
    random.seed(random_seed)

def save_checkpoint(checkpoint_dict, save_dir, save_file):
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, save_file)
    torch.save(checkpoint_dict, output_path)

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint_dict = torch.load(checkpoint_path)
    
    start_epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    if scheduler:
        scheduler.load_state_dict(checkpoint_dict['scheduler'])  

    return start_epoch, model, optimizer, scheduler  

class Log:
    def __init__(self, args, mode):
        self.args = args
        self.mode = mode
  
        self.loss_sum = 0
        self.acc_sum = 0
        self.count = 0
        self.loss_avg = 0
        self.acc_avg = 0

        if mode == 'train':
            log_dir = os.path.join(args.log_dir, 'train')
        elif mode == 'val':
            log_dir = os.path.join(args.log_dir, 'val')
        elif mode == 'test':
            log_dir = os.path.join(args.log_dir, 'test')
        os.makedirs(log_dir, exist_ok=True)

        file_path = os.path.join(log_dir, f'{args.date}.txt')

        self.logfile = open(file_path, 'a')
    
    def init(self):
        self.loss_sum = 0
        self.acc_sum = 0
        self.count = 0
        self.loss_avg = 0
        self.acc_avg = 0
    
    def accumulate(self, loss, acc, N=1):
        self.loss_sum += loss * N
        self.acc_sum += acc * N
        self.count += N
        self.loss_avg = self.loss_sum / self.count
        self.acc_avg = self.acc_sum / self.count
    
    def write(self, epoch, iter, len_dataloader, **kwargs):
        if kwargs:
            print(', '.join(f'{key}: {value}' for key, value in kwargs.items()))
        else:
            print(f'Epoch: {epoch+1}/{self.args.n_epochs} || Iter: {(iter+1)//self.args.accumulate_iter}/{len_dataloader//self.args.accumulate_iter} || Training Loss: {round(self.loss_avg, 4)}, Training acc: {round(self.acc_avg, 4)}')
        sys.stdout.flush()

        if kwargs:
            self.logfile.write(', '.join(f'{key}: {value}' for key, value in kwargs.items()) + '\n')
        else:
            self.logfile.write(f'Epoch: {epoch+1}/{self.args.n_epochs} || Iter: {(iter+1)//self.args.accumulate_iter}/{len_dataloader//self.args.accumulate_iter} || Training Loss: {round(self.loss_avg, 4)}, Training acc: {round(self.acc_avg, 4)}' + '\n')
        self.logfile.flush()
        
    def __del__(self): # __del__ is called when object is de-allocated from memory
        self.logfile.close()

def IoU(pred, gt):
    # IoU between n_box pred-boxes and 1 gt-box
    # pred : shape (N, n_box, 4) where 4 : x_c, y_c, w, h
    # gt : shape (N, 1, 4) where 4 : x_tl, y_tl, w, h
    
    gt = gt.repeat(1, pred.shape[1], 1) # shape (N, 1, 4) -> shape (N, n_box, 4) 

    # 4 : x_tl, ytl, x_br, y_br (tl : top-left, br : bottom-right)
    pred_box = torch.stack([pred[:, :, 0]-pred[:, :, 2]/2, pred[:, :, 1]-pred[:, :, 3]/2, pred[:, :, 0]+pred[:, :, 2]/2, pred[:, :, 1]+pred[:, :, 3]/2], dim=2) # shape (N, n_box, 4)
    # 4 : x_tl, ytl, x_br, y_br
    gt_box = torch.stack([gt[:, :, 0], gt[:, :, 1], gt[:, :, 0]+gt[:, :, 2], gt[:, :, 1]+gt[:, :, 3]], dim=2) # shape (N, n_box, 4)

    x_tl = torch.maximum(pred_box[:, :, 0], gt_box[:, :, 0]) # shape (N, n_box) 
    y_tl = torch.maximum(pred_box[:, :, 1], gt_box[:, :, 1])
    x_br = torch.minimum(pred_box[:, :, 2], gt_box[:, :, 2]) 
    y_br = torch.minimum(pred_box[:, :, 3], gt_box[:, :, 3])
    
    intersect = (x_br - x_tl) * (y_br - y_tl) # shape (N, n_box)
    intersect[x_br < x_tl] = 0
    intersect[y_br < y_tl] = 0

    pred_area = (pred_box[:, :, 2] - pred_box[:, :, 0]) * (pred_box[:, :, 3] - pred_box[:, :, 1]) # shape (N, n_box)
    gt_area = (gt_box[:, :, 2] - gt_box[:, :, 0]) * (gt_box[:, :, 3] - gt_box[:, :, 1])
    union = pred_area + gt_area - intersect
    
    iou = intersect / (union + 1e-6)

    return iou # iou : shape (N, n_box)

def IoU2(box1, box2):
    # box1 : shape (4,) where 4 : x_c, y_c, w, h
    # box2 : shape (4,) where 4 : x_c, y_c, w, h

    x_tl = box1[0] - box1[2]/2 if box1[0] - box1[2]/2 > box2[0] - box2[2]/2 else box2[0] - box2[2]/2
    y_tl = box1[1] - box1[3]/2 if box1[1] - box1[3]/2 > box2[1] - box2[3]/2 else box2[1] - box2[3]/2
    x_br = box1[0] + box1[2]/2 if box1[0] + box1[2]/2 < box2[0] + box2[2]/2 else box2[0] + box2[2]/2
    y_br = box1[1] + box1[3]/2 if box1[1] + box1[3]/2 < box2[1] + box2[3]/2 else box2[1] + box2[3]/2

    intersect = (x_br - x_tl) * (y_br - y_tl) if x_br > x_tl and y_br > y_tl else 0

    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - intersect

    iou = intersect / (union + 1e-6)

    return iou # scala