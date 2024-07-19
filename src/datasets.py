import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from src.utils import transform, label_map


class DefectsDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, keep_difficult=False):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'TEST', 'TEMPLATES'}

        self.data_folder = data_folder
        self.keep_difficult = keep_difficult

        self.images = []
        self.objects = []

        # Iterate through each dataset folder
        # for dataset_folder in os.listdir(data_folder):
        # dataset_path = os.path.join(data_folder, dataset_folder)
        dataset_path = data_folder
        if os.path.isdir(dataset_path):
            bbox_folder = os.path.join(dataset_path, 'bboxes')
            train_folder = os.path.join(dataset_path, 'train')

            for bbox_file in os.listdir(bbox_folder):
                if bbox_file.endswith('.txt'):
                    image_file = os.path.join(train_folder, bbox_file.replace('.txt', '.jpg'))
                    if not os.path.isfile(image_file):
                        image_file = os.path.join(train_folder, bbox_file.replace('.txt', '.png'))
                    if not os.path.isfile(image_file):
                        image_file = os.path.join(train_folder, bbox_file.replace('.txt', '.bmp'))

                    if os.path.isfile(image_file):
                        self.images.append(image_file)
                        self.objects.append(self.parse_annotation(os.path.join(bbox_folder, bbox_file)))

    def parse_annotation(self, annotation_path):
        boxes = []
        labels = []
        difficulties = []
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                class_name = parts[0]
                x1, y1, x2, y2 = map(int, parts[1:])
                boxes.append([x1, y1, x2, y2])
                labels.append(label_map[class_name])
                difficulties.append(0)  # Assuming difficulties are not specified in the provided format
        return {'boxes': boxes, 'labels': labels, 'difficulties': difficulties}

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels, difficulties)
        objects = self.objects[i]
        boxes = torch.FloatTensor(objects['boxes'])  # (n_objects, 4)
        labels = torch.LongTensor(objects['labels'])  # (n_objects)
        difficulties = torch.ByteTensor(objects['difficulties'])  # (n_objects)

        # Discard difficult objects, if desired
        if not self.keep_difficult:
            boxes = boxes[1 - difficulties]
            labels = labels[1 - difficulties]
            difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=self.split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        boxes = list()
        labels = list()
        difficulties = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])

        images = torch.stack(images, dim=0)
        return images, boxes, labels, difficulties  # tensor (N, 3, 300, 300), 3 lists of N tensors each
