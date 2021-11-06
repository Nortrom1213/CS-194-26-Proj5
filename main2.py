import os
from skimage import io, transform
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from PIL import Image
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

class FaceLandmarksDataset(Dataset):

    def __init__(self, root_dir, length, transform=None):
        self.root_dir = root_dir
        self.length=length
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx//6+1 in [8,12,14,15,22,30,35]:
          gender = 'f'
        else:
          gender = 'm'
        img_name = os.path.join(self.root_dir,'{:02d}-{:d}{}.jpg'.format(idx//6+1,idx%6+1,gender))
        image = io.imread(img_name)
        file = open(self.root_dir + '{:02d}-{:d}{}.asf'.format(idx//6+1,idx%6+1,gender))
        points = file.readlines()[16:74]
        landmarks = []
        for point in points:
          x,y = point.split('\t')[2:4]
          landmarks.append([float(x), float(y)])
        sample = {'image': image, 'landmarks': np.array(landmarks).astype('float32')}

        sample['image'] = Image.fromarray(sample['image'])
        bright = random.uniform(0.5,2.5)
        sample['image'] = transforms.functional.adjust_brightness(sample['image'] , bright)
        sample['image'] = np.array(sample['image'])

        if self.transform:
            sample = self.transform(sample)


        image = rgb2gray(sample['image'])
        image = image.astype('float32')-0.5
        sample['image'] = torch.from_numpy(image)
        return sample


class Rescale(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        img = transform.resize(image, (self.output_size[0], \
                                       self.output_size[1]))

        return {'image': img, 'landmarks': landmarks}


class Rotate(object):

    def __init__(self, max_angle):
        self.max_angle = max_angle

    def __call__(self, sample):
        seq = iaa.Sequential([iaa.Affine(rotate=(random.uniform(0, 1) - 0.5) * 2 * self.max_angle)])
        image, landmarks = sample['image'], sample['landmarks']
        landmarks[:, 0] = image.shape[1] * landmarks[:, 0]
        landmarks[:, 1] = image.shape[0] * landmarks[:, 1]
        kps = []
        for i in range(landmarks.shape[0]):
            kps.append(Keypoint(x=landmarks[i, 0], y=landmarks[i, 1]))
        kps = KeypointsOnImage(kps, shape=image.shape)
        image_aug, kps_aug = seq(image=image, keypoints=kps)
        for i in range(landmarks.shape[0]):
            landmarks[i, 0] = kps_aug.keypoints[i].x / image.shape[1]
            landmarks[i, 1] = kps_aug.keypoints[i].y / image.shape[0]
        return {'image': image_aug, 'landmarks': landmarks}


class ToTensor(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image = image
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(image.shape[1]*landmarks[:,0], image.shape[0]*landmarks[:,1], marker='.', c='r')
    plt.pause(0.001)


def show_landmarks_batch(sample_batched):
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['landmarks']
    grid = utils.make_grid(images_batch.unsqueeze(1))
    plt.imshow(grid.numpy().transpose((1, 2, 0))+0.5)
    plt.scatter(landmarks_batch[0,:,0].numpy()*images_batch.shape[2],
                    landmarks_batch[0,:,1].numpy()*images_batch.shape[1],
                    s=10, marker='.', c='r')


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 7)
        self.conv2 = nn.Conv2d(12, 18, 6)
        self.conv3 = nn.Conv2d(18, 24, 5)
        self.conv4 = nn.Conv2d(24, 32, 4)
        self.conv5 = nn.Conv2d(32, 42, 3)
        self.fc1 = nn.Linear(42*6*11, 400)
        self.fc2 = nn.Linear(400, 58*2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def getAccuracy(y1, y2):
    acc = 0.0
    y1 = y1.cpu().detach().numpy()
    y2 = y2.cpu().detach().numpy()
    for i in range(len(y1)):
        area1 = min(y1[i][0], y2[i][0])*min(y1[i][1], y2[i][1])
        area2 = max(y1[i][0], y2[i][0])*max(y1[i][1], y2[i][1])
        acc = acc + area1/area2
    return acc/(len(y1))




trans = transforms.Compose([Rescale((120,160)),
                            Rotate(15),
                            ToTensor()])

face_dataset = FaceLandmarksDataset(root_dir='../data/', length=240, transform=trans)

dataloader = DataLoader(face_dataset, batch_size=1, shuffle=False, num_workers=2)


# Part 2.1
for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    if i_batch % 6 == 1:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.savefig("Sample Picture for Prob 2 " + str(i_batch // 6))
        if i_batch == 25:
            break


# Part 2.2
net = Net().float()
print(net)

PATH = 'prob2_50epoch.pth'

train = 1
if train:
    epoch_num = 50
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    final = np.zeros((epoch_num, 2))
    for epoch in range(epoch_num):  # loop over the dataset multiple times

        running_loss = 0.0
        valid_loss = 0.0
        for i, data in enumerate(dataloader):
            if i < 192:
                inputs, labels = \
                    data['image'], data['landmarks']
                # zero the parameter gradients
                optimizer.zero_grad()

                outputs = net(inputs.unsqueeze(1).float())
                loss = criterion(outputs.view(-1, 58, 2), labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            else:
                inputs, labels = \
                    data['image'], data['landmarks']
                outputs = net(inputs.unsqueeze(1).float())
                loss = criterion(outputs.view(-1, 58, 2), labels)
                valid_loss += loss.item()
        final[epoch, 0] = running_loss / 192
        final[epoch, 1] = valid_loss / 48
    print('Finished Training')

    t1 = np.arange(0.0, epoch_num, 1.0)
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(12, 9))
    plt.plot(t1, final[:, 0], 'b', t1, final[:, 1], 'k')
    plt.legend(['Training', 'Validation'])
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.title('Loss curve')
    plt.savefig("loss_curve_prob_2.jpg")

    torch.save(net.state_dict(), PATH)
else:
    net = Net()
    net.load_state_dict(torch.load(PATH))


#Part 2.3
best_two = []
worst_two = []
best_two_samples = []
worst_two_samples = []
net.eval()


for i in range(len(face_dataset)):
    sample = face_dataset[i]
    image = sample['image']
    marks = sample['landmarks']
    x_val = image
    y_val = marks
    outputs = net(x_val.unsqueeze(0).unsqueeze(0).float())
    prediction = outputs.cpu().data.numpy()
    acc = getAccuracy(outputs, y_val.reshape(1, 116))
    if len(best_two_samples) < 2:
        best_two_samples.append({'image': image, 'landmarks': marks, 'pre': prediction.reshape(58, 2)})
        worst_two_samples.append({'image': image, 'landmarks': marks, 'pre': prediction.reshape(58, 2)})
        best_two.append(acc)
        worst_two.append(acc)
    else:
        if(acc > min(best_two)):
            index_ = best_two.index(min(best_two))
            best_two[index_] = acc
            best_two_samples[index_] = {'image': image, 'landmarks': marks, 'pre': prediction.reshape(58, 2)}
        if(acc < max(worst_two)):
            index_ = worst_two.index(min(worst_two))
            worst_two[index_] = acc
            worst_two_samples[index_] = {'image': image, 'landmarks': marks, 'pre': prediction.reshape(58, 2)}



i = 0
for sample in best_two_samples:
    i += 1
    plt.figure()
    image = sample['image']
    marks = sample['landmarks']
    print(marks)
    prediction = sample['pre']
    plt.imshow(image, cmap='gray')
    plt.scatter(image.shape[1]*marks[:, 0], image.shape[0]*marks[:, 1], s=150, marker='.', c='r')
    plt.scatter(image.shape[1]*prediction[:, 0], image.shape[0]*prediction[:, 1], s=150, marker='.', c='green')
    plt.savefig("Prob2_good" + str(i) + ".jpg")

i = 0
for sample in worst_two_samples:
    i += 1
    plt.figure()
    image = sample['image']
    marks = sample['landmarks']
    prediction = sample['pre']
    plt.imshow(image, cmap='gray')
    plt.scatter(image.shape[1]*marks[:, 0], image.shape[0]*marks[:, 1], s=150, marker='.', c='r')
    plt.scatter(image.shape[1]*prediction[:, 0], image.shape[0]*prediction[:, 1], s=150, marker='.', c='green')
    plt.savefig("Prob2_bad" + str(i) + ".jpg")


# Part 2.4
i = 0
for layer in net.state_dict():
    print(layer)

plt.imshow(net.state_dict()['conv5.weight'][8].numpy()[2, :, :])