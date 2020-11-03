import torch.utils.data as data
from .datasets import pkummd as pk
from .reader import FrameReader


class collate_fn:
    def __call__(self, batch):
        # print("Batch length: and type of ", len(batch), type(batch[0]))
        return batch[0][0], batch[0][1]


def get_data_generator(opt, transform):

    if opt.dataset == "THUMOS2014":
        dataset = th.THUMOS2014(opt, transform=transform)
    elif opt.dataset == "PKUMMD":
        dataset = pk.PKUMMD(opt, transform=transform)
    else:
        print("No implementation for the dataset {}".format(opt.dataset))

    # data_loader = data.DataLoader(dataset, batch_size=1,
    #   shuffle=False, pin_memory=False, num_workers=opt.workers, collate_fn=collate_fn())
    data_loader = dataset

    frame_reader = FrameReader(data_loader)
    frame_reader.start()
    return frame_reader
