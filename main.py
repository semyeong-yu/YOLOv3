from importlib import import_module
from utils import *
import torch
import torch.nn as nn
from tqdm import tqdm
from data import *
from data import _collate_fn
from torch.utils.data import DataLoader
from train import Runner
from datetime import datetime
from model import MyYOLO
import pudb

def train(args, id_convert_dict):
    # initialize model, optimizer, and scheduler
    model = MyYOLO(hidden_dim=32, n_class=args.n_class) # COCO dataset has 80 classes
    optimizer = getattr(import_module("torch.optim"), args.optimizer)(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    scheduler = getattr(import_module("torch.optim.lr_scheduler"), args.scheduler)(optimizer, mode='min', patience=5, verbose=True)
    start_epoch = 0

    # load checkpoint
    if not args.no_resume:
        start_epoch, model, optimizer, scheduler = load_checkpoint(args.checkpoint_train, model, optimizer, scheduler)
    
    # dataset
    train_dataset = CustomDataset(args, 'train', id_convert_dict)
    val_dataset = CustomDataset(args, 'val', id_convert_dict)

    # dataloader
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, collate_fn=_collate_fn, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, collate_fn=_collate_fn, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # initialize runner, and log
    runner = Runner(args=args, model=model, optimizer=optimizer, scheduler=scheduler)
    train_log = Log(args, 'train')
    val_log = Log(args, 'val')
    train_log.write(start_epoch, 0, 0, start_epoch=start_epoch, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    val_log.write(start_epoch, 0, 0, start_epoch=start_epoch, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    # best val acc
    best_acc = 0

    # train
    for epoch in range(start_epoch, args.n_epochs):
        
        optimizer.zero_grad()

        train_acc, train_loss = runner.train(train_loader, train_log, epoch)

        if (epoch+1) % args.val_epoch == 0 or epoch == args.n_epochs - 1:
            val_acc, val_loss = runner.validate(val_loader, val_log, epoch)

            save_checkpoint(
                {
                    'epoch' : epoch,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, os.path.join(args.checkpoint_dir, args.date), f"{epoch}.pt"
            )
            
            save_checkpoint(
                {
                    'epoch' : epoch,
                    'model' : model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict()
                }, os.path.join(args.checkpoint_dir, args.date), "latest.pt"
            )

            if (best_acc < val_acc):
                best_acc = val_acc
                
                save_checkpoint(
                    {
                        'epoch' : epoch,
                        'model' : model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict()
                    }, os.path.join(args.checkpoint_dir, args.date), "best.pt"
                )

        torch.cuda.empty_cache()

def test(args, id_convert_dict):
    class_name_dict = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        4: "airplane",
        5: "bus",
        6: "train",
        7: "truck",
        8: "boat",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        12: "parking meter",
        13: "bench",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
        24: "backpack",
        25: "umbrella",
        26: "handbag",
        27: "tie",
        28: "suitcase",
        29: "frisbee",
        30: "skis",
        31: "snowboard",
        32: "sports ball",
        33: "kite",
        34: "baseball bat",
        35: "baseball glove",
        36: "skateboard",
        37: "surfboard",
        38: "tennis racket",
        39: "bottle",
        40: "wine glass",
        41: "cup",
        42: "fork",
        43: "knife",
        44: "spoon",
        45: "bowl",
        46: "banana",
        47: "apple",
        48: "sandwich",
        49: "orange",
        50: "broccoli",
        51: "carrot",
        52: "hot dog",
        53: "pizza",
        54: "donut",
        55: "cake",
        56: "chair",
        57: "couch",
        58: "potted plant",
        59: "bed",
        60: "dining table",
        61: "toilet",
        62: "tv",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        68: "microwave",
        69: "oven",
        70: "toaster",
        71: "sink",
        72: "refrigerator",
        73: "book",
        74: "clock",
        75: "vase",
        76: "scissors",
        77: "teddy bear",
        78: "hair drier",
        79: "toothbrush"
    }

    # initialize model, optimizer, and scheduler
    model = MyYOLO(hidden_dim=32, n_class=args.n_class) # COCO dataset has 80 classes

    # load checkpoint
    _, model, _, _ = load_checkpoint(args.checkpoint_best, model, None, None)
    
    # dataset
    test_dataset = CustomDataset(args, 'test', id_convert_dict)

    # dataloader
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, collate_fn=_collate_fn, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # initialize runner, and log
    runner = Runner(args=args, model=model)
    test_log = Log(args, 'test')
    test_log.write(0, 0, 0, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    runner.test(test_loader, class_name_dict)

def main(args):
    fix_seed(args.random_seed)

    args.date = datetime.now().strftime('%m%d')

    id_convert_dict = {
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7,
        9: 8,
        10: 9,
        11: 10,
        13: 11,
        14: 12,
        15: 13,
        16: 14,
        17: 15,
        18: 16,
        19: 17,
        20: 18,
        21: 19,
        22: 20,
        23: 21,
        24: 22,
        25: 23,
        27: 24,
        28: 25,
        31: 26,
        32: 27,
        33: 28,
        34: 29,
        35: 30,
        36: 31,
        37: 32,
        38: 33,
        39: 34,
        40: 35,
        41: 36,
        42: 37,
        43: 38,
        44: 39,
        46: 40,
        47: 41,
        48: 42,
        49: 43,
        50: 44,
        51: 45,
        52: 46,
        53: 47,
        54: 48,
        55: 49,
        56: 50,
        57: 51,
        58: 52,
        59: 53,
        60: 54,
        61: 55,
        62: 56,
        63: 57,
        64: 58,
        65: 59,
        67: 60,
        70: 61,
        72: 62,
        73: 63,
        74: 64,
        75: 65,
        76: 66,
        77: 67,
        78: 68,
        79: 69,
        80: 70,
        81: 71,
        82: 72,
        84: 73,
        85: 74,
        86: 75,
        87: 76,
        88: 77,
        89: 78,
        90: 79
    }

    if args.mode == 'train':
        train(args, id_convert_dict)
    elif args.mode == 'test':
        test(args, id_convert_dict)

# python main.py --mode train --resume False --n_epochs 10 --val_epoch 2
if __name__ == "__main__":
    args = arg_parse()
    main(args)