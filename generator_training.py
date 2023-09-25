## Import necessary libs
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models, transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
import time
import sys
import cv2
import os
from PIL import Image
from tempfile import TemporaryDirectory
from skimage.metrics import peak_signal_noise_ratio as psnr
import random
from tqdm import tqdm

## Define necessary functions and classes
def generate_mask_rgb(I, B, threshold=30):
    """ Generate a binary mask of a RGB image given.
        The RGB values should vary from 0-255.
        I - Input image, blurred by raindrops,
        B - Background image or Ground Truth mage.
    """

    # Convert 0-1 or 0-255
    if torch.max(I) <= 1.0:
        threshold = threshold / 256
    # Compute the absolute difference between the degraded image and clean image
    diff = torch.abs(I - B)
    # Convert difference to grayscale by taking the mean across color channels
    diff_gray = torch.mean(diff, dim=1, keepdim=True)
    # Apply threshold to generate the mask
    mask = (diff_gray >= threshold).float()
    return mask

def batch_psnr(gts, outputs, data_range=None):
    """ This function calculates SUM of PSNR for all images in a batch. """
    assert gts.shape == outputs.shape
    psnr_sum = 0.0
    for i in range(gts.shape[0]):
        gt = gts[i,:,:,:].cpu().data.numpy().transpose((1,2,0))*255
        output = outputs[i,:,:,:].cpu().data.numpy().transpose((1,2,0))*255
        psnr_sum += psnr(gt, output,data_range=data_range)
    return psnr_sum

def read_last_line(file_path):
    """ Read the last line of the file of the given file path"""

    with open(file_path, 'r') as f:
        lines = f.readlines()
        return lines[-1] if lines else None
    
def loss_plot(file_name):
    # Step 1: Read the log file
    with open(file_name, 'r') as f:
        lines = f.readlines()

    # Step 2: Extract Loss and PSNR values
    epochs = []
    losses = {x : [] for x in ['train', 'val']}
    PSNRs =  {x : [] for x in ['train', 'val']}

    for line in lines:
        line = line.strip()
        epoch, loss, psnr = line.split(' ')
        if 'train' in loss:
            epochs.append(int(epoch.split(':')[1]))
            losses['train'].append(float(loss.split(':')[1]))
            PSNRs['train'].append(float(psnr.split(':')[1]))
        else:
            losses['val'].append(float(loss.split(':')[1]))
            PSNRs['val'].append(float(psnr.split(':')[1]))

    # Step 3: Plot these values
    plt.figure(figsize=(12, 6))

    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses['train'], color='red', label='Train Loss')
    plt.plot(epochs, losses['val'], color='blue', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Epoch vs Loss')
    plt.legend()

    # Plotting PSNR
    plt.subplot(1, 2, 2)
    plt.plot(epochs, PSNRs['train'], color='red', label='Train PSNR')
    plt.plot(epochs, PSNRs['val'], color='blue', label='Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title('Epoch vs PSNR')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_loss_every_epoch(losses, PSNRs):
    fig, axes = plt.subplots(1, 2, figsize=(10,10))
    # loss
    axes[0].plot(losses['train'], label='Training Loss')
    axes[0].plot(losses['val'], label='Validation Loss')
    axes[0].set_title('Training and Validation Losses')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    #PSNR
    axes[1].plot(PSNRs['train'], label='Training PSNR')
    axes[1].plot(PSNRs['val'], label='Validation PSNR')
    axes[1].set_title('Training and Validation PSNRs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('PSNR')
    axes[1].legend()
    plt.show()

def patch_random_cropped(img, gt, height=256, width=256):
    """ Get a radnom height * width area in both img and gt """

    start_y = random.randint(0, img.shape[0] - height)
    start_x = random.randint(0, img.shape[1] - width)
    end_y = start_y + height
    end_x = start_x + width
    # Extract the 256x256 area from the image
    return img[start_y:end_y, start_x:end_x], gt[start_y:end_y, start_x:end_x]


class MyImageDataset(Dataset):
    """ A personalised Dataset class, contains corresponding blurred and clean image.
        Every time __getitem__ is called, a pair of images already cropped will be returned.
    """

    def __init__(self, input_dir, gt_dir, transform=None):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.transform = transform
        self.input_list = sorted(os.listdir(input_dir))
        self.gt_list = []
        for i in range(len(self.input_list)):
            input_file_name = self.input_list[i]
            if 'rain' in self.input_list[i]:
                gt_file_name = input_file_name.replace('rain', 'clean')
            elif 'right' in self.input_list[i]:
                gt_file_name = input_file_name.replace('right', 'left')
            else:
                print("ERROR")
            self.gt_list.append(gt_file_name)

    def __len__(self):
        return (len(self.input_list))

    def __getitem__(self,idx):
        input_path = self.input_dir + self.input_list[idx]
        gt_path = self.gt_dir + self.gt_list[idx]
        input = cv2.imread(input_path)
        gt = cv2.imread(gt_path)
        input, gt = patch_random_cropped(input, gt)
        if self.transform:
            input = self.transform(input)
            gt = self.transform(gt)
        return input, gt
    

#Set iteration time
ITERATION = 4

#Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.det_conv0 = nn.Sequential(
            nn.Conv2d(4, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.det_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
            )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
            )
        self.det_conv_mask = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            )
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, 5, 1, 2),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.diconv1 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 2, dilation = 2),
            nn.ReLU()
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 4, dilation = 4),
            nn.ReLU()
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 8, dilation = 8),
            nn.ReLU()
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 16, dilation = 16),
            nn.ReLU()
            )
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.conv10 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe1 = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.outframe2 = nn.Sequential(
            nn.Conv2d(128, 3, 3, 1, 1),
            nn.ReLU()
            )
        self.output = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
            )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        mask = Variable(torch.ones(batch_size, 1, row, col)).cuda() / 2.
        h = Variable(torch.zeros(batch_size, 32, row, col)).cuda() 
        c = Variable(torch.zeros(batch_size, 32, row, col)).cuda()
        mask_list = []
        for i in range(ITERATION):
            x = torch.cat((input, mask), 1)
            x = self.det_conv0(x)
            resx = x
            x = F.relu(self.det_conv1(x) + resx)
            resx = x
            x = F.relu(self.det_conv2(x) + resx)
            resx = x
            x = F.relu(self.det_conv3(x) + resx)
            resx = x
            x = F.relu(self.det_conv4(x) + resx)
            resx = x
            x = F.relu(self.det_conv5(x) + resx)
            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * F.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)
        x = torch.cat((input, mask), 1)
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        frame1 = self.outframe1(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        frame2 = self.outframe2(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)
        return mask_list, frame1, frame2, x
    

def trainable(net, trainable):
    for para in net.parameters():
        para.requires_grad = trainable

#Initialize VGG16 with pretrained weight on ImageNet
def vgg_init():
    vgg_model = torchvision.models.vgg16(pretrained = True).cuda()
    trainable(vgg_model, False)
    return vgg_model

#Extract features from internal layers for perceptual loss
class vgg(nn.Module):
    def __init__(self, vgg_model):
        super(vgg, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '1': "relu1_1",
            '3': "relu1_2",
            '6': "relu2_1",
            '8': "relu2_2"
        }

    def forward(self, x):
        output = []
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output.append(x)
        return output

    
## Define Loss Functions
def L_att(attention_maps, binary_mask, N=4, theta=0.8):
    sum  = 0.0
    binary_mask = binary_mask.float()
    loss = nn.MSELoss()
    for i in range(N):
        t = i + 1
        sum += theta**(N-t) * loss(attention_maps[i], binary_mask)
    return sum

def L_M(scales, gts, lambda_i=[0.6, 0.8, 1.0]):
    sum = 0.0
    loss = nn.MSELoss()
    for i in range(3):
        sum += lambda_i[i] * loss(scales[i], gts[i])
    return sum

def L_P(output, gt, vgg):
    mse_loss = nn.MSELoss()
    output_features = vgg(output)
    gt_features = vgg(gt)

    loss = 0.0
    for o, g in zip(output_features, gt_features):
        loss += mse_loss(o, g)
    return loss

## Training process
def train_model(dataloaders, data_sizes, generator, optimizer, scheduler, best_PSNR, start_epoch, 
                device, basic_path, model_name, num_epochs=70, vgg=None, 
                loss_choice={'att':True, 'M':True, 'P':True}):
    
    torch.cuda.empty_cache()
    epoch = start_epoch

    checkpoint_path = os.path.join(basic_path, 'checkpoint.pth')
    best_PSNR_log_path = os.path.join(basic_path, 'best_PSNR.txt')
    log_file_path = os.path.join(basic_path, model_name+'_log.txt')

    while epoch <= num_epochs:
        print(f'Epoch{epoch}/{num_epochs}')
        print('-' * 20)
        generator_weights_path = os.path.join(basic_path, 'weights', str(epoch) + '.pt')
        losses = {x : [] for x in ['train', 'val']}
        PSNRs =  {x : [] for x in ['train', 'val']}


        for phase in ['train', 'val']:
            running_loss = 0.0
            running_PSNR = 0.0
            # Iterate over data
            progress_bar = tqdm(dataloaders[phase], desc='Epoch {:03d}'.format(epoch),
                                leave=False, disable=False)
            for img, gt in progress_bar:
                img = img.to(device)
                gt  =  gt.to(device)
                binary_mask = generate_mask_rgb(img, gt)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase=='train'):
                    mask_list, frame1, frame2, x = generator(img)

                    scales = [frame1, frame2, x]
                    gts = [ F.interpolate(gt, size=(64,64), mode='bilinear', align_corners=True),
                            F.interpolate(gt, size=(128,128), mode='bilinear', align_corners=True),
                            gt]

                    loss_att = L_att(mask_list, binary_mask) if loss_choice['att'] else 0.0
                    loss_M = L_M(scales, gts) if loss_choice['M'] else 0.0
                    loss_P = L_P(x, gt, vgg)if loss_choice['P'] else 0.0
                    loss = loss_att + loss_M + loss_P

                    # backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                if phase == 'val':
                    mask_list, frame1, frame2, x = generator(img)

                    scales = [frame1, frame2, x]
                    gts = [ F.interpolate(gt, size=(64,64), mode='bilinear', align_corners=True),
                            F.interpolate(gt, size=(128,128), mode='bilinear', align_corners=True),
                            gt]

                    loss_att = L_att(mask_list, binary_mask) if loss_choice['att'] else 0.0
                    loss_M = L_M(scales, gts) if loss_choice['M'] else 0.0
                    loss_P = L_P(x, gt, vgg)if loss_choice['P'] else 0.0
                    loss = loss_att + loss_M + loss_P

                # statistics
                running_loss += loss.item()
                running_PSNR += batch_psnr(gt, x, 256)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / data_sizes[phase]
            epoch_PSNR = running_PSNR / data_sizes[phase]
            losses[phase].append(epoch_loss)
            PSNRs[phase].append(epoch_PSNR)

            print(f'{phase} Loss: {epoch_loss:.6f}, PSNR: {epoch_PSNR:.6f}')
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'Epoch:{epoch} {phase}Loss:{epoch_loss} {phase}PSNR:{epoch_PSNR}\n')

        # best PSNR
        if epoch_PSNR > best_PSNR:
            best_PSNR = epoch_PSNR
            # torch.save(generator.state_dict(), generator_weights_path)
            #print("Best model saved at epoch", epoch)
            with open(best_PSNR_log_path, 'w') as log_file:
                log_file.write(str(best_PSNR))

        checkpoint = {
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'epoch': epoch
        }
        # save checkpoint and model every epoch
        torch.save(checkpoint, checkpoint_path)
        torch.save(generator.state_dict(), generator_weights_path)
        # plot every epoch to see where to end
        #loss_plot(log_file_path)
        epoch += 1

    print(f'Best PSNR: {best_PSNR:4f}')

def data_preparation(batch_size=10):

    data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ])}   
    train_set = MyImageDataset(input_dir = '/content/drive/MyDrive/dataset/train/data/',
                                gt_dir = '/content/drive/MyDrive/dataset/train/gt/',
                                transform = data_transforms['train'])
    
    val_set = MyImageDataset(input_dir = '/content/drive/MyDrive/dataset/val/data/',
                                gt_dir = '/content/drive/MyDrive/dataset/val/gt/',
                                transform = data_transforms['val'])
    
    image_datasets = {'train': train_set,
                        'val': val_set}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                    for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    return dataloaders, dataset_sizes

def run(basic_path, model_name, loss_choice):
    model_path = os.path.join(basic_path, model_name)
    checkpoint_path = os.path.join(basic_path, 'checkpoint.pth')
    best_PSNR_log_path = os.path.join(basic_path, 'best_PSNR.txt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    vgg_16 = vgg(vgg_init())
    vgg_16 = vgg_16.to(device)

    N_EPOCHS = 70
    BATCH_SIZE = 8
    LR_INIT = 1e-7
    POWER = 0.2
    optimizer = torch.optim.Adam(generator.parameters(), lr=LR_INIT)
    scheduler = PolynomialLR(optimizer,total_iters=N_EPOCHS,power=POWER)

    if 'checkpoint.pth' not in os.listdir(basic_path):
        # New Training. Set all the parameters.
        # Load model from Qian's parameters
        initial_weights_path = '/content/drive/MyDrive/DeRaindrop-master/weights/gen.pkl'
        generator.load_state_dict(torch.load(initial_weights_path))
        start_epoch = 1
        best_PSNR = 0

    else:
        # Continue Training. Load all the parameters from the checkpoint.
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1  # you'll start from the next epoch
        best_PSNR = float(read_last_line(best_PSNR_log_path))
    
    dataloaders, data_sizes = data_preparation(BATCH_SIZE)
    train_model(dataloaders, data_sizes, generator, optimizer, scheduler, best_PSNR, start_epoch, device,
                basic_path, model_name, num_epochs=N_EPOCHS, vgg=vgg_16, loss_choice=loss_choice)
