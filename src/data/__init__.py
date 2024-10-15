from .cocowholebody import WholeBodyDataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist


def build_dataset(args):
    if args.dataset == "wholebody":
        dataset_train = WholeBodyDataset(args, "train")
        dataset_val = WholeBodyDataset(args, "val")

    return dataset_train, dataset_val


def build_dataloader(args, dataset_train, dataset_val):
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train,
                                        shuffle=True,
                                        drop_last=True)
        sampler_val = DistributedSampler(dataset_val,
                                        shuffle=False,
                                        drop_last=True)
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.data["samples_per_gpu"],
            sampler=sampler_train,
            num_workers=args.num_workers,
            pin_memory=True
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.data["samples_per_gpu"],
            sampler=sampler_val,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.data["samples_per_gpu"],
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.data["samples_per_gpu"],
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

    return dataloader_train, dataloader_val
