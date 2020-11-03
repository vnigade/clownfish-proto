import torch
import torch.utils.data as data
from PIL import Image
import os
import functools
import json
import copy
import math


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):

    class_labels_map = {}
    for _, value in data['database'].items():
        label = int(value['annotations']['label'])
        label_name = str(value['annotations']['label_name'])
        class_labels_map[label_name] = label

    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            if subset == 'testing':
                video_names.append('{}'.format(key))
            else:
                video_names.append('{}'.format(key))
                annotations.append(value['annotations'])

    return video_names, annotations


def get_video_from_clip(video_name):
    return video_name.split("-clip")[0]


def modify_frame_indices(video_dir_path, frame_indices):
    modified_indices = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if not os.path.exists(image_path):
            return modified_indices
        modified_indices.append(i)
    return modified_indices


def get_untrimmed_label(video, start_frame, end_frame):
    label = []
    label_names = []
    for clip in video:
        action = clip['label']
        label_name = clip['label_name']
        start = clip['segment'][0]
        end = clip['segment'][1]
        for frame in range(start_frame, end_frame, 1):
            if frame in list(range(start, end)):
                label.append(action)
                label_names.append(label_name)
    return label, label_names


def make_untrimmed_dataset(root_path, scores_dump_path, annotation_path, subset, window_size, window_stride):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    videos = {}
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        annotation = annotations[i]
        label = int(annotation['label'])
        label_name = annotation['label_name']
        begin_t = annotation['start_frame']
        end_t = annotation['end_frame']
        clip = {
            'label': label,
            'label_name': label_name,
            'segment': [begin_t, end_t],
            'video_id': video_names[i]
        }
        video = get_video_from_clip(video_names[i])
        if not video in videos:
            videos[video] = []

        videos[video].append(clip)

    i = 0
    # dump_dir = os.path.join(root_path, scores_dump_path)
    for video in videos:
        video_path = os.path.join(root_path, video)
        if not os.path.exists(video_path) or os.path.exists(scores_dump_path + "/" + video):
            print("Skipping video ", video)
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        window_start = 1
        window_end = window_start + window_size
        idx = 1
        while window_end < n_frames:
            sample = {
                'video': video_path,
                'segment': [window_start, window_end],
                'video_id': video,
                'window_id': idx}
            frame_indices = list(range(window_start, window_end))
            frame_indices = modify_frame_indices(
                sample['video'], frame_indices)
            sample['frame_indices'] = frame_indices
            # sample['label'] = -1  # Not computed yet
            sample['label'], sample['label_name'] = get_untrimmed_label(
                videos[video], window_start, window_end)
            if len(frame_indices) == window_size:
                dataset.append(sample)

            window_start += window_stride
            window_end = window_start + window_size
            idx += 1
        if i % 10 == 0:
            print('dataset loading [{}/{}]'.format(i, len(videos)))
            # break early exit
        i += 1

    print("Make untrimmed dataset")
    return dataset, idx_to_class


class PKUMMD(data.Dataset):
    def __init__(self,
                 opt,
                 transform=None,
                 get_loader=get_default_video_loader,
                 scores_dump_path=""):

        window_size = opt.window_size
        window_stride = opt.window_stride
        self.data, self.class_names = make_untrimmed_dataset(
            opt.dataset_path, scores_dump_path, opt.dataset_annotation, "validation",
            window_size, window_stride)

        self.transform = transform
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        video_data = self.loader(path, frame_indices)[0]
        if self.transform is not None:
            video_data = self.transform(video_data)
        video_id = self.data[index]['video_id']
        label = self.data[index]['label']
        label_name = self.data[index]['label_name']
        start_frame = frame_indices[0]
        end_frame = frame_indices[-1]
        video_meta = {'video_id': video_id, 'start_frame': start_frame,
                      'end_frame': end_frame, 'label': label, 'label_name': label_name}
        return video_meta, video_data

    def __len__(self):
        return len(self.data)
