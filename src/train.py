import os
import tqdm
import numpy as np
import argparse

import torch
import torchvision
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import transforms

from data.dataset import Dataset
from model.hkudetector import resnet50
from utils.loss import yololoss

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('--yolo_S', default=14, type=int, help='YOLO grid num')  # 14
parser.add_argument('--yolo_B', default=4, type=int, help='YOLO box num')
parser.add_argument('--yolo_C', default=5, type=int, help='detection class num')

parser.add_argument('--num_epochs', default=30, type=int, help='number of epochs')  # 10
parser.add_argument('--batch_size', default=8, type=int, help='batch size')  # 12
parser.add_argument('--learning_rate', default=5e-5, type=float, help='learning rate')  # 0.00001

parser.add_argument('--seed', default=666, type=int, help='random seed')  #666
parser.add_argument('--dataset_root', default='./ass1_dataset', type=str, help='dataset root')  # './ass1_dataset'
parser.add_argument('--output_dir', default='checkpoints', type=str, help='output directory')

parser.add_argument('--l_coord', default=5., type=float, help='hyper parameter for localization loss')
parser.add_argument('--l_noobj', default=0.5, type=float, help='hyper parameter for no object loss')

args = parser.parse_args()


def load_pretrained(net):
    resnet = torchvision.models.resnet50(pretrained=True)
    resnet_state_dict = resnet.state_dict()

    net_dict = net.state_dict()
    for k in resnet_state_dict.keys():
        if k in net_dict.keys() and not k.startswith('fc'):
            net_dict[k] = resnet_state_dict[k]
    net.load_state_dict(net_dict)


def draw_lines(epochs, tl, vl, algorithm='resNet'):
    x1 = np.arange(0, epochs)
    y1 = tl
    y2 = vl
    plt.subplot(2, 1, 1)
    plt.plot(x1+1, y1, label='Train Loss')
    plt.plot(x1+1, y2, label='Valid Loss')
    plt.legend()
    plt.xlabel('epoches')
    plt.title('Loss')
    plt.tight_layout()
    plt.savefig('./Result for ' + algorithm)
    plt.show()
    """
    x1 = np.arange(0, epochs)
    y1 = tl
    y2 = vl
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, label='Train Loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.title('TLoss')
    plt.subplot(2, 1, 2)
    plt.plot(x1, y2, label='Valid Loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.title('VLoss')
    plt.tight_layout()
    plt.savefig('./Result for ' + algorithm)
    plt.show()
    """


def initial_list():
    tl = []
    vl = []
    return tl, vl


####################################################################
# Environment Setting
# We suggest using only one GPU, or you should change the codes about model saving and loading

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('NUMBER OF CUDA DEVICES:', torch.cuda.device_count())

# Other settings
args.load_pretrain = True
print(args)

output_dir = args.output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.random.seed(args.seed)
torch.manual_seed(args.seed)

####################################################################
criterion = yololoss(args, l_coord=args.l_coord, l_noobj=args.l_noobj)

hku_mmdetector = resnet50(args=args)
if args.load_pretrain:
    load_pretrained(hku_mmdetector)
hku_mmdetector = hku_mmdetector.to(device)

####################################################################
# Multiple GPUs if needed
# if torch.cuda.device_count() > 1:
#     hku_mmdetector = torch.nn.DataParallel(hku_mmdetector)

hku_mmdetector.train()

# initialize optimizer
optimizer = torch.optim.AdamW(hku_mmdetector.parameters(), betas=(0.9, 0.999), lr=args.learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=4)
#  scheduler = ReduceLROnPlateau(optimizer, 'min')
#  optimizer = torch.optim.SGD(hku_mmdetector.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.45)  #  5e-4

# initialize dataset
train_dataset = Dataset(args, split='train', transform=[transforms.ToTensor()])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

###################################################################
# Please fill the codes below to initialize the validation dataset
##################################################################
val_dataset = Dataset(args, split='val', transform=[transforms.ToTensor()])
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
##################################################################

print(f'NUMBER OF DATA SAMPLES: {len(train_dataset)}')
print(f'BATCH SIZE: {args.batch_size}')

train_lossList, val_lossList = initial_list()

train_dict = dict(iter=[], loss=[])
best_val_loss = np.inf

for epoch in range(args.num_epochs):
    hku_mmdetector.train()

    # training
    total_loss = 0.
    print(('\n' + '%10s' * 3) % ('epoch', 'loss', 'gpu'))
    progress_bar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader))
    for i, (images, target) in progress_bar:
        images = images.to(device)
        target = target.to(device)

        pred = hku_mmdetector(images)  # (bs, S, S, B*5+C)
        loss = criterion(pred, target)

        total_loss += loss.data

        ###################################################################
        # Please fill the codes here to complete the gradient backward
        ##################################################################
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##################################################################

        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        s = ('%10s' + '%10.4g' + '%10s') % ('%g/%g' % (epoch + 1, args.num_epochs), total_loss / (i + 1), mem)
        progress_bar.set_description(s)

    train_lossList.append(total_loss / len(train_loader))

    # validation
    validation_loss = 0.0
    hku_mmdetector.eval()
    progress_bar = tqdm.tqdm(enumerate(val_loader), total=len(val_loader))
    with torch.no_grad():
        for i, (images, target) in progress_bar:
            images = images.to(device)
            target = target.to(device)

            prediction = hku_mmdetector(images)
            loss = criterion(prediction, target)
            validation_loss += loss.data
    validation_loss /= len(val_loader)
    val_lossList.append(validation_loss)
    print("validation loss:", validation_loss.item())
    scheduler.step()

    if best_val_loss > validation_loss:
        best_val_loss = validation_loss

        save = {'state_dict': hku_mmdetector.state_dict()}
        torch.save(save, os.path.join(output_dir, 'hku_mmdetector_best.pth'))

    save = {'state_dict': hku_mmdetector.state_dict()}
    torch.save(save, os.path.join(output_dir, 'hku_mmdetector_epoch_' + str(epoch + 1) + '.pth'))

    torch.cuda.empty_cache()

draw_lines(args.num_epochs, train_lossList, val_lossList, 'resNet')
