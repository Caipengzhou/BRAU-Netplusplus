import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'
import platform
import pathlib
plt = platform.system()
if plt != 'Windows':
  pathlib.WindowsPath = pathlib.PosixPath
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import optim
import random
import time
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from loader import  binary_class
from sklearn.model_selection import GroupKFold
from loss import *
# from synapse_train_test.networks.bra_unet import BRAUnet
from networks.bra_unet import BRAUnet
def get_train_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.25),
            A.VerticalFlip(p=0.25),
            A.ShiftScaleRotate(shift_limit=0, p=0.25),
            A.CoarseDropout(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])


def get_valid_transform():
    return A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensor()
        ])


def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}

    best_model_wts = model.state_dict()
    best_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = []
            running_corrects = []

            # Iterate over data
            # for inputs,labels,label_for_ce,image_id in dataloaders[phase]:
            for inputs, labels, image_id in dataloaders[phase]:
                # wrap them in Variable
                if torch.cuda.is_available():

                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                    # label_for_ce = Variable(label_for_ce.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                score = accuracy_metric(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # calculate loss and IoU
                running_loss.append(loss.item())
                running_corrects.append(score.item())

            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_corrects)

            print('{} Loss: {:.4f} IoU: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)

            # save parameters
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                counter = 0
                if epoch > 0:
                    torch.save(model.state_dict(), f'save_models/epoch_{epoch}_{epoch_acc}.pth')
            elif phase == 'valid' and epoch_loss > best_loss:
                counter += 1
            if phase == 'train':
                print('Current learning rate:', optimizer.param_groups[0]['lr'])
                scheduler.step()
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, Loss_list, Accuracy_list


if __name__ == '__main__':


    seed = 1234
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/home/cqut/Data/medical_seg_data/ISIC2018_jpg/', help='the path of images')
    parser.add_argument('--csvfile', type=str, default='src/test_train_data.csv',
                        help='two columns [image_id,category(train/test)]')
    parser.add_argument('--loss', default='dice', help='loss type')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--epoch', type=int, default=200, help='epoches')
    args = parser.parse_args()

    os.makedirs(f'save_models/', exist_ok=True)

    df = pd.read_csv(args.csvfile)
    df = df[df.category == 'train']
    df.reset_index(drop=True, inplace=True)
    gkf = GroupKFold(n_splits=5)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df.image_id.tolist())):
        df.loc[val_idx, 'fold'] = fold

    fold = 0
    val_files = list(df[df.fold == fold].image_id)
    print(val_files)
    print(len(val_files))
    train_files = list(df[df.fold != fold].image_id)
    print(train_files)
    print(len(train_files))

    train_dataset = binary_class(args.dataset, train_files, get_train_transform())
    val_dataset = binary_class(args.dataset, val_files, get_valid_transform())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,
                                               drop_last=True,num_workers=8)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch//4, drop_last=True,num_workers=8)

    dataloaders = {'train': train_loader, 'valid': val_loader}

    model_ft = BRAUnet(img_size=256,in_chans=3, num_classes=1, n_win=8)
    model_ft.load_from()

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()

    # Loss, IoU and Optimizer
    if args.loss == 'ce':
        criterion = nn.BCELoss()
    if args.loss == 'dice':
        criterion = DiceLoss_binary()
    accuracy_metric = IoU_binary()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, 200, eta_min=0, last_epoch=-1, verbose=False)
    model_ft, Loss_list, Accuracy_list = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                                                     num_epochs=args.epoch)
    torch.save(model_ft.state_dict(), f'save_models/epoch_last.pth')




