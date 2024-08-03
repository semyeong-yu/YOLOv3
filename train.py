from tqdm import tqdm
import torch
import torch.nn as nn
from utils import *
import einops
import cv2

class Runner:
    def __init__(self, args, model, optimizer=None, scheduler=None):

        self.dtype = torch.float32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def train(self, dataloader, train_log, epoch):
        
        self.model.train()
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        for iter, (x, y_gt) in pbar:
            # x : shape (N, C, H, W) range [0., 1.]
            # y_gt : {'label' : tensor (N, max_n_box), 'bbox' : tensor (N, max_n_box, 4)}
            x = x.cuda(non_blocking=True)
            y_label = y_gt['label'].cuda(non_blocking=True) # shape (N, max_n_box)
            y_bbox = y_gt['bbox'].cuda(non_blocking=True) # shape (N, max_n_box, 4)

            # forward
            model = self.model.to(self.device)
            feat_1, feat_2, feat_3 = model(x)
            # feat_1, feat_2, feat_3 : shape (N, 255, 52, 52), (N, 255, 26, 26), (N, 255, 13, 13)
            # 255 = 3 anchors * (4 bboxes + 1 objectness + 80 classes)
            
            feat_1_bbox = torch.cat([feat_1[:, :4, :, :], feat_1[:, 85:89, :, :], feat_1[:, 170:174, :, :]], dim=1) # shape (N, 12, 52, 52)
            init_feat_1_bbox_offset = einops.rearrange(feat_1_bbox, 'n (n_anchor box) h w -> n (n_anchor h w) box', box=4) # shape (N, n_box, 4) where n_box = 3 * 52 * 52
            feat_1_objectness = torch.stack([feat_1[:, 4, :, :], feat_1[:, 89, :, :], feat_1[:, 174, :, :]], dim=1) # shape (N, 3, 52, 52)
            init_feat_1_objectness = einops.rearrange(feat_1_objectness, 'n n_anchor h w -> n (n_anchor h w)') # shape (N, n_box) where n_box = 3 * 52 * 52
            feat_1_class = torch.cat([feat_1[:, 5:85, :, :], feat_1[:, 90:170, :, :], feat_1[:, 175:, :, :]], dim=1) # shape (N, 240, 52, 52)
            init_feat_1_class = einops.rearrange(feat_1_class, 'n (n_anchor n_class) h w -> n (n_anchor h w) n_class', n_class=self.args.n_class) # shape (N, n_box, 80) where n_box = 3 * 52 * 52
            
            feat_2_bbox = torch.cat([feat_2[:, :4, :, :], feat_2[:, 85:89, :, :], feat_2[:, 170:174, :, :]], dim=1) # shape (N, 12, 26, 26)
            init_feat_2_bbox_offset = einops.rearrange(feat_2_bbox, 'n (n_anchor box) h w -> n (n_anchor h w) box', box=4) # shape (N, 12, 26, 26) -> (N, n_box, 4) where n_box = 3 * 26 * 26
            feat_2_objectness = torch.stack([feat_2[:, 4, :, :], feat_2[:, 89, :, :], feat_2[:, 174, :, :]], dim=1) # shape (N, 3, 26, 26)
            init_feat_2_objectness = einops.rearrange(feat_2_objectness, 'n n_anchor h w -> n (n_anchor h w)') # shape (N, n_box) where n_box = 3 * 26 * 26
            feat_2_class = torch.cat([feat_2[:, 5:85, :, :], feat_2[:, 90:170, :, :], feat_2[:, 175:, :, :]], dim=1) # shape (N, 240, 26, 26)
            init_feat_2_class = einops.rearrange(feat_2_class, 'n (n_anchor n_class) h w -> n (n_anchor h w) n_class', n_class=self.args.n_class) # shape (N, n_box, 80) where n_box = 3 * 26 * 26

            feat_3_bbox = torch.cat([feat_3[:, :4, :, :], feat_3[:, 85:89, :, :], feat_3[:, 170:174, :, :]], dim=1) # shape (N, 12, 13, 13)
            init_feat_3_bbox_offset = einops.rearrange(feat_3_bbox, 'n (n_anchor box) h w -> n (n_anchor h w) box', box=4) # shape (N, 12, 13, 13) -> (N, n_box, 4) where n_box = 3 * 13 * 13
            feat_3_objectness = torch.stack([feat_3[:, 4, :, :], feat_3[:, 89, :, :], feat_3[:, 174, :, :]], dim=1) # shape (N, 3, 13, 13)
            init_feat_3_objectness = einops.rearrange(feat_3_objectness, 'n n_anchor h w -> n (n_anchor h w)') # shape (N, n_box) where n_box = 3 * 13 * 13
            feat_3_class = torch.cat([feat_3[:, 5:85, :, :], feat_3[:, 90:170, :, :], feat_3[:, 175:, :, :]], dim=1) # shape (N, 240, 13, 13)
            init_feat_3_class = einops.rearrange(feat_3_class, 'n (n_anchor n_class) h w -> n (n_anchor h w) n_class', n_class=self.args.n_class) # shape (N, n_box, 80) where n_box = 3 * 13 * 13

            loss = 0
            loss_n = 0
            acc = 0
            acc_n = 0

            for idx in range(y_label.shape[1]): # iterate for each gt bbox
                box_i = y_bbox[:, idx, :] # idx-th gt bbox : shape (N, 4)
                label_i = y_label[:, idx] # idx-th gt label : shape (N,)

                # filter-out if there is NO GT BOX.(padded)
                exist_gt = label_i != -1 # shape (N,) e.g. [True, True, False, True, False]
                box_i = box_i[exist_gt] # shape (new_N, 4)
                label_i = label_i[exist_gt] # shape (new_N,)
                feat_1_bbox_offset = init_feat_1_bbox_offset[exist_gt]
                feat_1_objectness = init_feat_1_objectness[exist_gt]
                feat_1_class = init_feat_1_class[exist_gt]
                feat_2_bbox_offset = init_feat_2_bbox_offset[exist_gt]
                feat_2_objectness = init_feat_2_objectness[exist_gt]
                feat_2_class = init_feat_2_class[exist_gt]
                feat_3_bbox_offset = init_feat_3_bbox_offset[exist_gt]
                feat_3_objectness = init_feat_3_objectness[exist_gt]
                feat_3_class = init_feat_3_class[exist_gt]

                '''
                x : 416 * 416 [pixel]
                feat_1 : 52 * 52 [grid]
                1 [grid] corresponds to 8 * 8 [pixel]
                '''
                # get IoU
                feat_1_bbox_bbox = self.offset_to_bbox(x.shape[2], x.shape[3], feat_1.shape[2], feat_1.shape[3], feat_1_bbox_offset) # convert from offset to box for IoU calculation # shape (N, n_box, 4)
                small_iou = IoU(feat_1_bbox_bbox, box_i.unsqueeze(1)) # shape (N, n_box)
                # get loss
                loss_small = self.loss(small_iou, feat_1_bbox_offset, feat_1_objectness, feat_1_class, box_i, label_i, feat_1.shape[2], feat_1.shape[3])    
                acc_small = self.accuracy(small_iou, feat_1_objectness, feat_1_class, label_i)

                '''
                x : 416 * 416 [pixel]
                feat_2 : 26 * 26 [grid]
                1 [grid] corresponds to 16 * 16 [pixel]
                '''
                # get IoU
                feat_2_bbox_bbox = self.offset_to_bbox(x.shape[2], x.shape[3], feat_2.shape[2], feat_2.shape[3], feat_2_bbox_offset) # convert from offset to box for IoU calculation # shape (N, n_box, 4)
                middle_iou = IoU(feat_2_bbox_bbox, box_i.unsqueeze(1)) # shape (N, n_box)
                # get loss
                loss_middle = self.loss(middle_iou, feat_2_bbox_offset, feat_2_objectness, feat_2_class, box_i, label_i, feat_2.shape[2], feat_2.shape[3])    
                acc_middle = self.accuracy(middle_iou, feat_2_objectness, feat_2_class, label_i)

                '''
                x : 416 * 416 [pixel]
                feat_3 : 13 * 13 [grid]
                1 [grid] corresponds to 32 * 32 [pixel]
                '''
                # get IoU
                feat_3_bbox_bbox = self.offset_to_bbox(x.shape[2], x.shape[3], feat_3.shape[2], feat_3.shape[3], feat_3_bbox_offset) # convert from offset to box for IoU calculation # shape (N, n_box, 4)
                big_iou = IoU(feat_3_bbox_bbox, box_i.unsqueeze(1)) # shape (N, n_box)
                # get loss
                loss_big = self.loss(big_iou, feat_3_bbox_offset, feat_3_objectness, feat_3_class, box_i, label_i, feat_3.shape[2], feat_3.shape[3])  
                acc_big = self.accuracy(big_iou, feat_3_objectness, feat_3_class, label_i)

                loss = (loss_n * loss + (loss_small + loss_middle + loss_big)) / (loss_n + 1)
                acc = (acc_n * acc + (acc_small + acc_middle + acc_big) / 3) / (acc_n + 1)
                loss_n += 1
                acc_n += 1
            
            with torch.no_grad():
                train_log.accumulate(loss.item(), acc, x.size(0)) # 더해서
                # use train_log.loss_avg # 나눔
            
            loss = loss / self.args.accumulate_iter # 나눠서
            loss.backward() # 더함

            # accumulate, and then backward
            if (iter + 1) % self.args.accumulate_iter == 0:
                # gradient clipping
                nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                # backward
                self.optimizer.step()
                self.scheduler.step(loss.item())

                self.optimizer.zero_grad()

                # tqdm log
                description = f'Epoch: {epoch+1}/{self.args.n_epochs} || Iter: {(iter+1)//self.args.accumulate_iter}/{len(dataloader)//self.args.accumulate_iter} || Training Loss: {round(train_log.loss_avg, 4)}, Training acc: {round(train_log.acc_avg, 4)}'
                pbar.set_description(description)

                # log
                train_log.write(epoch, iter, len(dataloader))
                if iter != len(dataloader) - 1:
                    train_log.init()

            torch.cuda.empty_cache()
        
        return train_log.acc_avg, train_log.loss_avg

    def validate(self, dataloader, val_log, epoch):
        
        self.model.eval()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        with torch.no_grad():
            for iter, (x, y_gt) in pbar:
                # x : shape (N, C, H, W) range [0., 1.]
                # y_gt : {'label' : tensor (N, max_n_box), 'bbox' : tensor (N, max_n_box, 4)}
                x = x.cuda(non_blocking=True)
                y_label = y_gt['label'].cuda(non_blocking=True) # shape (N, max_n_box)
                y_bbox = y_gt['bbox'].cuda(non_blocking=True) # shape (N, max_n_box, 4)

                # forward
                model = self.model.to(self.device)
                feat_1, feat_2, feat_3 = model(x)
                # feat_1, feat_2, feat_3 : shape (N, 255, 52, 52), (N, 255, 26, 26), (N, 255, 13, 13)
                # 255 = 3 anchors * (4 bboxes + 1 objectness + 80 classes)
                
                feat_1_bbox = torch.cat([feat_1[:, :4, :, :], feat_1[:, 85:89, :, :], feat_1[:, 170:174, :, :]], dim=1) # shape (N, 12, 52, 52)
                init_feat_1_bbox_offset = einops.rearrange(feat_1_bbox, 'n (n_anchor box) h w -> n (n_anchor h w) box', box=4) # shape (N, n_box, 4) where n_box = 3 * 52 * 52
                feat_1_objectness = torch.stack([feat_1[:, 4, :, :], feat_1[:, 89, :, :], feat_1[:, 174, :, :]], dim=1) # shape (N, 3, 52, 52)
                init_feat_1_objectness = einops.rearrange(feat_1_objectness, 'n n_anchor h w -> n (n_anchor h w)') # shape (N, n_box) where n_box = 3 * 52 * 52
                feat_1_class = torch.cat([feat_1[:, 5:85, :, :], feat_1[:, 90:170, :, :], feat_1[:, 175:, :, :]], dim=1) # shape (N, 240, 52, 52)
                init_feat_1_class = einops.rearrange(feat_1_class, 'n (n_anchor n_class) h w -> n (n_anchor h w) n_class', n_class=self.args.n_class) # shape (N, n_box, 80) where n_box = 3 * 52 * 52
                
                feat_2_bbox = torch.cat([feat_2[:, :4, :, :], feat_2[:, 85:89, :, :], feat_2[:, 170:174, :, :]], dim=1) # shape (N, 12, 26, 26)
                init_feat_2_bbox_offset = einops.rearrange(feat_2_bbox, 'n (n_anchor box) h w -> n (n_anchor h w) box', box=4) # shape (N, 12, 26, 26) -> (N, n_box, 4) where n_box = 3 * 26 * 26
                feat_2_objectness = torch.stack([feat_2[:, 4, :, :], feat_2[:, 89, :, :], feat_2[:, 174, :, :]], dim=1) # shape (N, 3, 26, 26)
                init_feat_2_objectness = einops.rearrange(feat_2_objectness, 'n n_anchor h w -> n (n_anchor h w)') # shape (N, n_box) where n_box = 3 * 26 * 26
                feat_2_class = torch.cat([feat_2[:, 5:85, :, :], feat_2[:, 90:170, :, :], feat_2[:, 175:, :, :]], dim=1) # shape (N, 240, 26, 26)
                init_feat_2_class = einops.rearrange(feat_2_class, 'n (n_anchor n_class) h w -> n (n_anchor h w) n_class', n_class=self.args.n_class) # shape (N, n_box, 80) where n_box = 3 * 26 * 26

                feat_3_bbox = torch.cat([feat_3[:, :4, :, :], feat_3[:, 85:89, :, :], feat_3[:, 170:174, :, :]], dim=1) # shape (N, 12, 13, 13)
                init_feat_3_bbox_offset = einops.rearrange(feat_3_bbox, 'n (n_anchor box) h w -> n (n_anchor h w) box', box=4) # shape (N, 12, 13, 13) -> (N, n_box, 4) where n_box = 3 * 13 * 13
                feat_3_objectness = torch.stack([feat_3[:, 4, :, :], feat_3[:, 89, :, :], feat_3[:, 174, :, :]], dim=1) # shape (N, 3, 13, 13)
                init_feat_3_objectness = einops.rearrange(feat_3_objectness, 'n n_anchor h w -> n (n_anchor h w)') # shape (N, n_box) where n_box = 3 * 13 * 13
                feat_3_class = torch.cat([feat_3[:, 5:85, :, :], feat_3[:, 90:170, :, :], feat_3[:, 175:, :, :]], dim=1) # shape (N, 240, 13, 13)
                init_feat_3_class = einops.rearrange(feat_3_class, 'n (n_anchor n_class) h w -> n (n_anchor h w) n_class', n_class=self.args.n_class) # shape (N, n_box, 80) where n_box = 3 * 13 * 13

                loss = 0
                loss_n = 0
                acc = 0
                acc_n = 0

                for idx in range(y_label.shape[1]): # iterate for each gt bbox
                    box_i = y_bbox[:, idx, :] # idx-th gt bbox : shape (N, 4)
                    label_i = y_label[:, idx] # idx-th gt label : shape (N,)

                    # filter-out if there is NO GT BOX.(padded)
                    exist_gt = label_i != -1 # shape (N,) e.g. [True, True, False, True, False]
                    box_i = box_i[exist_gt] # shape (new_N, 4)
                    label_i = label_i[exist_gt] # shape (new_N,)
                    feat_1_bbox_offset = init_feat_1_bbox_offset[exist_gt]
                    feat_1_objectness = init_feat_1_objectness[exist_gt]
                    feat_1_class = init_feat_1_class[exist_gt]
                    feat_2_bbox_offset = init_feat_2_bbox_offset[exist_gt]
                    feat_2_objectness = init_feat_2_objectness[exist_gt]
                    feat_2_class = init_feat_2_class[exist_gt]
                    feat_3_bbox_offset = init_feat_3_bbox_offset[exist_gt]
                    feat_3_objectness = init_feat_3_objectness[exist_gt]
                    feat_3_class = init_feat_3_class[exist_gt]

                    '''
                    x : 416 * 416 [pixel]
                    feat_1 : 52 * 52 [grid]
                    1 [grid] corresponds to 8 * 8 [pixel]
                    '''
                    # get IoU
                    feat_1_bbox_bbox = self.offset_to_bbox(x.shape[2], x.shape[3], feat_1.shape[2], feat_1.shape[3], feat_1_bbox_offset) # convert from offset to box for IoU calculation # shape (N, n_box, 4)
                    small_iou = IoU(feat_1_bbox_bbox, box_i.unsqueeze(1)) # shape (N, n_box)
                    # get loss
                    loss_small = self.loss(small_iou, feat_1_bbox_offset, feat_1_objectness, feat_1_class, box_i, label_i, feat_1.shape[2], feat_1.shape[3])    
                    acc_small = self.accuracy(small_iou, feat_1_objectness, feat_1_class, label_i)

                    '''
                    x : 416 * 416 [pixel]
                    feat_2 : 26 * 26 [grid]
                    1 [grid] corresponds to 16 * 16 [pixel]
                    '''
                    # get IoU
                    feat_2_bbox_bbox = self.offset_to_bbox(x.shape[2], x.shape[3], feat_2.shape[2], feat_2.shape[3], feat_2_bbox_offset) # convert from offset to box for IoU calculation # shape (N, n_box, 4)
                    middle_iou = IoU(feat_2_bbox_bbox, box_i.unsqueeze(1)) # shape (N, n_box)
                    # get loss
                    loss_middle = self.loss(middle_iou, feat_2_bbox_offset, feat_2_objectness, feat_2_class, box_i, label_i, feat_2.shape[2], feat_2.shape[3])    
                    acc_middle = self.accuracy(middle_iou, feat_2_objectness, feat_2_class, label_i)

                    '''
                    x : 416 * 416 [pixel]
                    feat_3 : 13 * 13 [grid]
                    1 [grid] corresponds to 32 * 32 [pixel]
                    '''
                    # get IoU
                    feat_3_bbox_bbox = self.offset_to_bbox(x.shape[2], x.shape[3], feat_3.shape[2], feat_3.shape[3], feat_3_bbox_offset) # convert from offset to box for IoU calculation # shape (N, n_box, 4)
                    big_iou = IoU(feat_3_bbox_bbox, box_i.unsqueeze(1)) # shape (N, n_box)
                    # get loss
                    loss_big = self.loss(big_iou, feat_3_bbox_offset, feat_3_objectness, feat_3_class, box_i, label_i, feat_3.shape[2], feat_3.shape[3])  
                    acc_big = self.accuracy(big_iou, feat_3_objectness, feat_3_class, label_i)

                    loss = (loss_n * loss + (loss_small + loss_middle + loss_big)) / (loss_n + 1)
                    acc = (acc_n * acc + (acc_small + acc_middle + acc_big) / 3) / (acc_n + 1)
                    acc_n += 1
                    loss_n += 1
                
                val_log.accumulate(loss.item(), acc, x.size(0))
                
                # accumulate, and then backward
                if (iter + 1) % self.args.accumulate_iter == 0:
                    # tqdm log
                    description = f'Epoch: {epoch+1}/{self.args.n_epochs} || Iter: {(iter+1)//self.args.accumulate_iter}/{len(dataloader)//self.args.accumulate_iter} || Training Loss: {round(val_log.loss_avg, 4)}, Training acc: {round(val_log.acc_avg, 4)}'
                    pbar.set_description(description)

                    # log
                    val_log.write(epoch, iter, len(dataloader))
                    if iter != len(dataloader) - 1:
                        val_log.init()

                torch.cuda.empty_cache()
        
        return val_log.acc_avg, val_log.loss_avg

    def test(self, dataloader, class_name_dict):
        self.model.eval()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        with torch.no_grad():
            for iter, (x, y_gt) in pbar:
                # x : shape (N, C, H, W) range [0., 1.]
                # y_gt : {}
                x = x.cuda(non_blocking=True)

                # forward
                model = self.model.to(self.device)
                feat_1, feat_2, feat_3 = model(x)
                # feat_1, feat_2, feat_3 : shape (N, 255, 52, 52), (N, 255, 26, 26), (N, 255, 13, 13)
                # 255 = 3 anchors * (4 bboxes + 1 objectness + 80 classes)
                
                feat_1_bbox = torch.cat([feat_1[:, :4, :, :], feat_1[:, 85:89, :, :], feat_1[:, 170:174, :, :]], dim=1) # shape (N, 12, 52, 52)
                feat_1_bbox_offset = einops.rearrange(feat_1_bbox, 'n (n_anchor box) h w -> n (n_anchor h w) box', box=4) # shape (N, n_box, 4) where n_box = 3 * 52 * 52
                feat_1_objectness = torch.stack([feat_1[:, 4, :, :], feat_1[:, 89, :, :], feat_1[:, 174, :, :]], dim=1) # shape (N, 3, 52, 52)
                feat_1_objectness = einops.rearrange(feat_1_objectness, 'n n_anchor h w -> n (n_anchor h w)') # shape (N, n_box) where n_box = 3 * 52 * 52
                feat_1_class = torch.cat([feat_1[:, 5:85, :, :], feat_1[:, 90:170, :, :], feat_1[:, 175:, :, :]], dim=1) # shape (N, 240, 52, 52)
                feat_1_class = einops.rearrange(feat_1_class, 'n (n_anchor n_class) h w -> n (n_anchor h w) n_class', n_class=self.args.n_class) # shape (N, n_box, 80) where n_box = 3 * 52 * 52
                
                feat_2_bbox = torch.cat([feat_2[:, :4, :, :], feat_2[:, 85:89, :, :], feat_2[:, 170:174, :, :]], dim=1) # shape (N, 12, 26, 26)
                feat_2_bbox_offset = einops.rearrange(feat_2_bbox, 'n (n_anchor box) h w -> n (n_anchor h w) box', box=4) # shape (N, 12, 26, 26) -> (N, n_box, 4) where n_box = 3 * 26 * 26
                feat_2_objectness = torch.stack([feat_2[:, 4, :, :], feat_2[:, 89, :, :], feat_2[:, 174, :, :]], dim=1) # shape (N, 3, 26, 26)
                feat_2_objectness = einops.rearrange(feat_2_objectness, 'n n_anchor h w -> n (n_anchor h w)') # shape (N, n_box) where n_box = 3 * 26 * 26
                feat_2_class = torch.cat([feat_2[:, 5:85, :, :], feat_2[:, 90:170, :, :], feat_2[:, 175:, :, :]], dim=1) # shape (N, 240, 26, 26)
                feat_2_class = einops.rearrange(feat_2_class, 'n (n_anchor n_class) h w -> n (n_anchor h w) n_class', n_class=self.args.n_class) # shape (N, n_box, 80) where n_box = 3 * 26 * 26

                feat_3_bbox = torch.cat([feat_3[:, :4, :, :], feat_3[:, 85:89, :, :], feat_3[:, 170:174, :, :]], dim=1) # shape (N, 12, 13, 13)
                feat_3_bbox_offset = einops.rearrange(feat_3_bbox, 'n (n_anchor box) h w -> n (n_anchor h w) box', box=4) # shape (N, 12, 13, 13) -> (N, n_box, 4) where n_box = 3 * 13 * 13
                feat_3_objectness = torch.stack([feat_3[:, 4, :, :], feat_3[:, 89, :, :], feat_3[:, 174, :, :]], dim=1) # shape (N, 3, 13, 13)
                feat_3_objectness = einops.rearrange(feat_3_objectness, 'n n_anchor h w -> n (n_anchor h w)') # shape (N, n_box) where n_box = 3 * 13 * 13
                feat_3_class = torch.cat([feat_3[:, 5:85, :, :], feat_3[:, 90:170, :, :], feat_3[:, 175:, :, :]], dim=1) # shape (N, 240, 13, 13)
                feat_3_class = einops.rearrange(feat_3_class, 'n (n_anchor n_class) h w -> n (n_anchor h w) n_class', n_class=self.args.n_class) # shape (N, n_box, 80) where n_box = 3 * 13 * 13

                '''
                x : 416 * 416 [pixel]
                feat_1 : 52 * 52 [grid]
                1 [grid] corresponds to 8 * 8 [pixel]
                '''
                feat_1_bbox_bbox = self.offset_to_bbox(x.shape[2], x.shape[3], feat_1.shape[2], feat_1.shape[3], feat_1_bbox_offset) # shape (N, n_box, 4)
                feat_1_objectness = torch.sigmoid(feat_1_objectness) # shape (N, n_box) where n_box = 3 * 52 * 52
                feat_1_class = torch.sigmoid(feat_1_class) # shape (N, n_box, 80) where n_box = 3 * 52 * 52

                '''
                x : 416 * 416 [pixel]
                feat_2 : 26 * 26 [grid]
                1 [grid] corresponds to 16 * 16 [pixel]
                '''
                feat_2_bbox_bbox = self.offset_to_bbox(x.shape[2], x.shape[3], feat_2.shape[2], feat_2.shape[3], feat_2_bbox_offset) # shape (N, n_box, 4)
                feat_2_objectness = torch.sigmoid(feat_2_objectness) # shape (N, n_box) where n_box = 3 * 26 * 26
                feat_2_class = torch.sigmoid(feat_2_class) # shape (N, n_box, 80) where n_box = 3 * 26 * 26

                '''
                x : 416 * 416 [pixel]
                feat_3 : 13 * 13 [grid]
                1 [grid] corresponds to 32 * 32 [pixel]
                '''
                feat_3_bbox_bbox = self.offset_to_bbox(x.shape[2], x.shape[3], feat_3.shape[2], feat_3.shape[3], feat_3_bbox_offset) # shape (N, n_box, 4)
                feat_3_objectness = torch.sigmoid(feat_3_objectness) # shape (N, n_box) where n_box = 3 * 13 * 13
                feat_3_class = torch.sigmoid(feat_3_class) # shape (N, n_box, 80) where n_box = 3 * 13 * 13

                # aggregate all boxes
                feat_bbox = torch.cat([feat_1_bbox_bbox, feat_2_bbox_bbox, feat_3_bbox_bbox], dim=1) # shape (N, n_box, 4) where n_box = 3*52*52 + 3*26*26 + 3*13*13
                feat_objectness = torch.cat([feat_1_objectness, feat_2_objectness, feat_3_objectness], dim=1) # shape (N, n_box) where n_box = 3*52*52 + 3*26*26 + 3*13*13
                feat_class = torch.cat([feat_1_class, feat_2_class, feat_3_class], dim=1) # shape (N, n_box, 80) where n_box = 3*52*52 + 3*26*26 + 3*13*13

                x = x.cpu()
                for i in range(x.shape[0]): # for each image
                    # NMS (Non-Maximum Suppression)
                    final_idx = self.NMS(feat_bbox[i], feat_objectness[i]) # list of chosen index
                    cv_bbox = feat_bbox[i, final_idx, :] # shape (new_n_box, 4)
                    cv_objectness = feat_objectness[i, final_idx] # shape (new_n_box,)
                    cv_class = feat_class[i, final_idx, :] # shape (new_n_box, 80)
                    
                    cv_img = x[i] # tensor (C, H, W) range[0., 1.]
                    cv_img = (cv_img.permute((1, 2, 0)).numpy() * 255).astype(int) # np.ndarray (H, W, C) [0, 255]

                    for j in range(cv_bbox.shape[0]): # for each bbox
                        x1 = int(cv_bbox[j][0] - cv_bbox[j][2] / 2) # x_tl
                        x2 = int(cv_bbox[j][0] + cv_bbox[j][2] / 2) # x_br
                        y1 = int(cv_bbox[j][1] - cv_bbox[j][3] / 2) # y_tl
                        y2 = int(cv_bbox[j][1] + cv_bbox[j][3] / 2) # y_br
                        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2) # green rectangle with thickness=2[px]
                        _, class_idx = torch.max(cv_class[j], dim=0, keepdim=False) # scala tensor(c) where c in range [0, 79]
                        label = class_name_dict[class_idx.item()]
                        cv2.putText(cv_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # blue text with font_size=0.5 at (x1, y1-10)
                        score = f"{cv_objectness[j]}"
                        cv2.putText(cv_img, score, (x2, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # blue text with font_size=0.5 and thickness=2[px] at (x2, y1-10)
                    
                    cv2.imshow(f"Image {i+1}", cv_img)
                    cv2.waitKey(0)  # wait for keyboard input

                torch.cuda.empty_cache()

    def offset_to_bbox(self, H, W, h, w, feat_offset):
        # H, W = x.shape[2], x.shape[3]
        # h, w = feat_1.shape[2], feat_1.shape[3]
        # feat_offset : shape (N, n_box, 4) where n_box = n_anchor * h * w = 3 * h * w
        '''
        c_x : [[0, 8, 16, ..., 408],
                [0, 8, 16, ..., 408],
                ...]
        c_y : [[0, 0, 0, ..., 0],
                [8, 8, 8, ..., 8],
                ...]
        '''
        c_x = torch.arange(0, W, W // w).repeat(h, 1) # shape (h, w) = (52, 52)
        c_y = torch.arange(0, H, H // h).reshape(h, 1).repeat(1, w) # shape (52, 52)
        p_w = torch.tensor(self.args.small_anchor_size[::2]) # shape (3,)
        p_h = torch.tensor(self.args.small_anchor_size[1::2]) # shape (3,)
        if self.device:
            c_x, c_y, p_w, p_h = c_x.to(self.device), c_y.to(self.device), p_w.to(self.device), p_h.to(self.device)

        N, _, coord = feat_offset.shape
        n_anchor = len(p_w)

        # shape (N, n_box, 4) -> shape (N, 3, 52, 52, 4)
        feat_offset = einops.rearrange(feat_offset, 'n (n_anchor h w) box -> n n_anchor h w box', n_anchor=n_anchor, h=h, w=w)
        # shape (52, 52) -> shape (N, 3, 52, 52)
        c_x = c_x.reshape(1, 1, h, w).repeat(N, n_anchor, 1, 1)
        # shape (52, 52) -> shape (N, 3, 52, 52)
        c_y = c_y.reshape(1, 1, h, w).repeat(N, n_anchor, 1, 1)
        # shape (3,) -> shape (N, 3, 52, 52)
        p_w = p_w.reshape(1, n_anchor, 1, 1).repeat(N, 1, h, w)
        # shape (3,) -> shape (N, 3, 52, 52)
        p_h = p_h.reshape(1, n_anchor, 1, 1).repeat(N, 1, h, w)

        # b_x = c_x + sigmoid(t_x)
        # b_y = c_y + sigmoid(t_y)
        # b_w = p_w * e^{t_w}
        # b_h = p_h * e^{t_h}
        b_x = c_x + torch.sigmoid(feat_offset[:, :, :, :, 0]) # shape (N, 3, 52, 52)
        b_y = c_y + torch.sigmoid(feat_offset[:, :, :, :, 1]) # shape (N, 3, 52, 52)
        b_w = p_w * torch.exp(feat_offset[:, :, :, :, 2]) # shape (N, 3, 52, 52)
        b_h = p_h * torch.exp(feat_offset[:, :, :, :, 3]) # shape (N, 3, 52, 52)

        return torch.stack([b_x, b_y, b_w, b_h], dim=-1).reshape(N, -1, coord) # shape (N, 3, 52, 52, 4) -> shape (N, 3*52*52, 4) = (N, n_box, 4) where 4 : x_c, y_c, w, h

    def bbox_to_offset(self, gt_box, anchor_idx):
        # gt_box : shape (N, 4) where 4 : x_tl, y_tl, w, h
        # anchor_idx : shape (N,) where element is 0 or 1 or 2
        N = gt_box.shape[0]
        gt_box = torch.stack([gt_box[:, 0]+gt_box[:, 2]//2, gt_box[:, 1]+gt_box[:, 3]//2, gt_box[:, 2], gt_box[:, 3]], dim=1) # shape (N, 4) where 4 : x_tl, y_tl, w, h -> x_c, y_c, w, h
        c_x = torch.floor(gt_box[:, 0]) # shape (N,)
        c_y = torch.floor(gt_box[:, 1]) # shape (N,)
        p_w = torch.where(anchor_idx == 0, torch.full((N,), self.args.small_anchor_size[0], dtype=int, device=self.device), anchor_idx) # shape (N,)
        p_w = torch.where(p_w == 1, torch.full((N,), self.args.small_anchor_size[2], dtype=int, device=self.device), p_w) # shape (N,)
        p_w = torch.where(p_w == 2, torch.full((N,), self.args.small_anchor_size[4], dtype=int, device=self.device), p_w) # shape (N,)
        p_h = torch.where(anchor_idx == 0, torch.full((N,), self.args.small_anchor_size[1], dtype=int, device=self.device), anchor_idx) # shape (N,)
        p_h = torch.where(p_h == 1, torch.full((N,), self.args.small_anchor_size[3], dtype=int, device=self.device), p_h) # shape (N,)
        p_h = torch.where(p_h == 2, torch.full((N,), self.args.small_anchor_size[5], dtype=int, device=self.device), p_h) # shape (N,)

        # t_x = inverse_sigmoid(b_x - c_x) = - log(1/(b_x - c_x) - 1)
        # t_y = inverse_sigmoid(b_y - c_y)
        # t_w = log(b_w / p_w)
        # t_h = log(b_h / p_h)
        t_x = - torch.log(1/(gt_box[:, 0] - c_x) - 1) # shape (N,)
        t_y = - torch.log(1/(gt_box[:, 1] - c_y) - 1) # shape (N,)
        t_w = torch.log(gt_box[:, 2] / p_w)
        t_h = torch.log(gt_box[:, 3] / p_h)

        return torch.stack([t_x, t_y, t_w, t_h], dim=1) # shape (N, 4)

    def loss(self, iou, feat_box, feat_objectness, feat_class, gt_box, gt_label, h, w):
        # iou : shape (N, n_box)
        # feat_box : shape (N, n_box, 4)
        # feat_objectness : shape (N, n_box)
        # feat_class : shape (N, n_box, 80)
        # gt_box : shape (N, 4)
        # gt_label : shape (N,)
        # h, w = feat_1.shape[2], feat_1.shape[3] = 52, 52
        N = iou.shape[0]

        # Step 1. find best-match feat_box
        _, best_box_idx = torch.max(iou, dim=1, keepdim=False) # shape (N,) and element range [0, n_box=3*52*52)
        best_box = torch.gather(feat_box, 1, best_box_idx.unsqueeze(1).unsqueeze(1).repeat(1, 1, feat_box.shape[2])).reshape(N, feat_box.shape[2]) # shape (N,) -> (N, 1, 1) -> (N, 1, 4) -> (N, 1, 4) after indexing (N, n_box, 4) -> (N, 4)
        best_objectness = torch.gather(feat_objectness, 1, best_box_idx.unsqueeze(1)).reshape(N) # shape (N,) -> (N, 1) -> (N, 1) after indexing (N, n_box) -> (N,)
        best_class = torch.gather(feat_class, 1, best_box_idx.unsqueeze(1).unsqueeze(1).repeat(1, 1, feat_class.shape[2])).reshape(N, feat_class.shape[2]) # shape (N,) -> (N, 1, 1) -> (N, 1, 80) -> (N, 1, 80) after indexing (N, n_box, 80) -> (N, 80)
        # Step 2. gt_box : self.bbox_to_offset()
        anchor_idx = best_box_idx // (h * w) # shape (N,) range [0, 3)
        gt_box_offset = self.bbox_to_offset(gt_box, anchor_idx) # shape (N, 4)
        # Step 3. calculate bbox, object, class loss
        criterion_BCE = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss after sigmoid
        loss_bbox = torch.sum((best_box - gt_box_offset) ** 2) # SSE b.w. best_box and gt_box_offset of shape (N, 4)
        loss_object_1 = criterion_BCE(best_objectness, torch.ones(best_objectness.shape, device=self.device)) # BCE b.w. best_object and 1 of shape (N,)
        one_hot_gt_label = torch.eye(feat_class.shape[2], device=self.device)[gt_label] # shape (N, 80)
        loss_class = criterion_BCE(best_class, one_hot_gt_label) # BCE b.w. best_class and one_hot_gt_label of shape (N, 80)

        # Step 1. find low-iou bbox
        low_iou = iou < self.args.iou_thres # shape (N, n_box)
        low_iou_box = feat_objectness[low_iou] # 1D-tensor after boolean-indexing
        # Step 2. calculate no-object (objectness=0) loss
        loss_object_2 = criterion_BCE(low_iou_box, torch.zeros(low_iou_box.shape, device=self.device)) # BCE b.w. low_iou_box and 0 of shape (n_low_iou_box,)
        
        loss = self.args.bbox_weight * loss_bbox + self.args.object_weight * (loss_object_1 + loss_object_2) + self.args.class_weight * loss_class

        return loss
    
    def accuracy(self, iou, feat_objectness, feat_class, gt_label, k=1):
        # iou : shape (N, n_box)
        # feat_objectness : shape (N, n_box)
        # feat_class : shape (N, n_box, 80)
        # gt_label : shape (N,)
        N = iou.shape[0]

        # Step 1. find best-match feat_box
        _, best_box_idx = torch.max(iou, dim=1, keepdim=False) # shape (N,) and element range [0, n_box)
        best_objectness = torch.gather(feat_objectness, 1, best_box_idx.unsqueeze(1)).reshape(N) # shape (N,) -> (N, 1) -> (N, 1) after indexing (N, n_box) -> (N,)
        best_class = torch.gather(feat_class, 1, best_box_idx.unsqueeze(1).unsqueeze(1).repeat(1, 1, feat_class.shape[2])).reshape(N, feat_class.shape[2]) # shape (N,) -> (N, 1, 1) -> (N, 1, 80) -> (N, 1, 80) after indexing (N, n_box, 80) -> (N, 80)
        
        best_class = best_class[best_objectness > 0.2] # shape (new_N, 80) # calculate accuracy only for objectness score > 0.2
        gt_label_obj = gt_label[best_objectness > 0.2] # shape (new_N,) # calculate accuracy only for objectness score > 0.2
        if gt_label_obj.numel() == 0:
            return 0
        _, best_class_idx = best_class.topk(k, dim=1) # shape (new_N, k)
        best_class_idx = best_class_idx if best_class_idx.shape[1] != 1 else best_class_idx.reshape(best_class.shape[0])

        n_correct = torch.sum(best_class_idx == gt_label_obj).item()

        return n_correct / len(gt_label_obj)
    
    def NMS(self, feat_bbox, feat_objectness):
        # feat_bbox : shape (n_box, 4) where n_box = 3*52*52 + 3*26*26 + 3*13*13
        # feat_objectness : shape (n_box,) where n_box = 3*52*52 + 3*26*26 + 3*13*13
        # where 4 : x_c, y_c, w, h

        final_idx = []
        
        for i in range(feat_bbox.shape[0]): # for each box
            discard = False
            for j in range(feat_bbox.shape[0]): # for each box
                if IoU2(feat_bbox[i], feat_bbox[j]) > self.args.NMS_thres:
                    if feat_objectness[i] < feat_objectness[j]:
                        discard = True
                        break
            
            if not discard:
                final_idx.append(i)

        return final_idx # list of length new_n_box with range [0, n_box)