import os
import torch
import glob
import cv2
import json

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, args, mode, id_convert_dict):
        super(CustomDataset, self).__init__()
        self.args = args
        self.mode = mode
        self.id_convert_dict = id_convert_dict

        # data path
        os.makedirs(args.data_dir, exist_ok=True)
        if mode == 'train':
            self.img_path = os.path.join(args.data_dir, args.img_dir_train)
            self.json_path = os.path.join(args.data_dir, args.gt_json_train)
        elif mode == 'val':
            self.img_path = os.path.join(args.data_dir, args.img_dir_val)
            self.json_path = os.path.join(args.data_dir, args.gt_json_val)
        elif mode == 'test':
            self.img_path = os.path.join(args.data_dir, args.img_dir_test)
            self.json_path = None

        if mode == 'train' or mode == 'val':        
            # read gt json
            if os.path.exists(self.json_path):
                with open(self.json_path, 'r') as f:
                    json_file = json.load(f)
                f.close()
            
            images = json_file['images'] # list of dicts in json
            annotations = json_file['annotations'] # list of dicts in json
            
            # Main Idea
            # img_id를 기준 삼아 make dict {img_id : ...}
            self.images = {} # dict {img_id : img_filename}
            self.annotations = {} # dict {img_id : list of annotations(dicts)} (since there might be multiple annotations(bboxes) in one image)

            for elem in annotations:
                if elem['image_id'] not in self.annotations: # key check
                    self.annotations[elem['image_id']] = [] # initialize a list for image_id key
                self.annotations[elem['image_id']].append(elem)

            for elem in images:
                if elem['id'] not in list(self.annotations.keys()): # filter-out images which have no gt info.
                    continue
                self.images[elem['id']] = elem['file_name']

            self.image_id = list(self.images.keys()) # list of unique img_id   # use for indexing at __getitem__()
        
        elif mode == 'test':
            # list of image filenames
            self.images = glob.glob(os.path.join(self.img_path, '*.jpg'))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # lazy loading
        if self.mode == 'train' or self.mode == 'val':
            img_id = self.image_id[idx] # idx-th image id
            
            # deal with image
            np_img = cv2.imread(os.path.join(self.img_path, self.images[img_id])) # np.ndarray (H, W, C) [0, 255]     
            img = torch.tensor(np_img.transpose((2, 0, 1)).astype(float)).mul_(1.0) / 255.0 # tensor (C, H, W) [0., 1.]

            # deal with annotation
            labels = []
            bboxes = []
            for annot_dict in self.annotations[img_id]:
                labels.append(self.id_convert_dict[annot_dict['category_id']]) # list of 0~79 for 80 classes
                bboxes.append(annot_dict['bbox']) # list of [x, y, w, h]

            labels = torch.tensor(labels) # tensor of shape (n_boxes,)
            bboxes = torch.tensor(bboxes) # tensor of shape (n_boxes, 4)

            # data augmentation
            if self.mode =='train':
                img, bboxes = self.augment(img, bboxes)

            target = {}
            target['label'] = labels
            target['bbox'] = bboxes
        
        elif self.mode == 'test':
            # deal with image
            np_img = cv2.imread(self.images[idx]) # np.ndarray (H, W, C) [0, 255]
            img = torch.tensor(np_img.transpose((2, 0, 1)).astype(float)).mul_(1.0) / 255.0 # tensor (C, H, W) [0., 1.]
            target = {} # empty target for test mode
        
        return img, target
    
    def augment(self, img, bboxes):
        # data augmentation
        # img : tensor (C, H, W) [0., 1.]
        # bboxes : tensor (n_boxes, 4)
        
        # Resize to tensor (C, self.args.crop_size[1], self.args.crop_size[0])
        h_ratio = img.shape[1] / self.args.crop_size[1]
        w_ratio = img.shape[2] / self.args.crop_size[0]

        img_o = torch.zeros(img.shape[0], self.args.crop_size[1], self.args.crop_size[0])
        bboxes_o = torch.zeros_like(bboxes)

        for i in range(img_o.shape[1]):
            for j in range(img_o.shape[2]):
                img_o[:, i, j] = img[:, int(i * h_ratio), int(j * w_ratio)]
                bboxes_o[:, [0, 2]] = bboxes[:, [0, 2]] / w_ratio
                bboxes_o[:, [1, 3]] = bboxes[:, [1, 3]] / h_ratio

        # RandomHorizontalFlip
        if torch.rand(1).item() < 0.5:
            img_oo = img_o.clone()
            for i in range(img_o.shape[2]):
                img_o[:, :, i] = img_oo[:, :, img_o.shape[2]-1 - i] # x = w-1 - x
            bboxes_o[:, 0] = img_o.shape[2]-1 - (bboxes_o[:, 0] + bboxes_o[:, 2]) # x = (w-1) - (x+w)

        # RandomVerticalFlip
        if torch.rand(1).item() < 0.5:
            img_oo = img_o.clone()
            for i in range(img_o.shape[1]):
                img_o[:, i, :] = img_oo[:, img_o.shape[1]-1 - i, :] # y = h-1 - y
            bboxes_o[:, 1] = img_o.shape[1]-1 - (bboxes_o[:, 1] + bboxes_o[:, 3]) # y = (h-1) - (y+h)
        
        return img_o, bboxes_o
    
def _collate_fn(samples):
    # samples : [(img1, target1), (img2, target2), ...] # tuple이 batch_size-개
    # img1 : tensor of shape (C, H, W) range [0., 1.] 
    # img1 :  CustomDataset().augment()로 (C, H, W) 맞춰줌
    # target1 : {'label' : tensor of shape (n_boxes,), 'bbox' : tensor of shape (n_boxes, 4)}
    # target1 : n_boxes가 image마다 다르므로 padding 필요

    img_list = []
    target_label_list = []
    target_bbox_list = []
    
    max_n_box = 0
    for elem in samples:
        if elem[1]: # train or val mode : elem[1] is not empty dict
            if elem[1]['label'].shape[0] > max_n_box:
                max_n_box = elem[1]['label'].shape[0]

    for elem in samples:
        img_list.append(elem[0])
        
        if elem[1]:
            target_label_pad = torch.full((max_n_box,), -1, dtype=int)
            target_label_pad[:elem[1]['label'].shape[0]] = elem[1]['label'] # shape (max_n_box,)
            target_label_list.append(target_label_pad)
            
            target_bbox_pad = torch.full((max_n_box, 4), -1, dtype=float)
            target_bbox_pad[:elem[1]['bbox'].shape[0], :] = elem[1]['bbox'] # shape (max_n_box, 4)
            target_bbox_list.append(target_bbox_pad)
    
    image = torch.stack(img_list, dim=0) # shape (N, C, H, W) range [0., 1.]
    
    if elem[1]:
        target_label = torch.stack(target_label_list, dim=0) # shape (N, max_n_box)
        target_bbox = torch.stack(target_bbox_list, dim=0) # shape (N, max_n_box, 4)
        target = {}
        target['label'] = target_label
        target['bbox'] = target_bbox
    else:
        target = {}

    return image, target
