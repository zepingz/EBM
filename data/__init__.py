from torch.utils.data import DataLoader
import torchvision.transforms as T

from .dummy import DummyDataset
from .moving_mnist import MovingMNISTDataset
from .poke import PokeDataset, PokeLinpredDataset


def add_data_specific_args(parser):
    # General
    parser.add_argument(
        "--dataset",
        default="moving_mnist",
        choices=["dummy", "moving_mnist", "poke"],
        help="Dataset type"
    )
    parser.add_argument(
        "--num_conditional_frames",
        default=2,
        type=int,
        help="Number of input conditional frames",
    )
    parser.add_argument("--num_workers", default=12, type=int)
    parser.add_argument("--batch_size", default=512, type=int)

    # Moving MNIST
    parser.add_argument(
        "--mnist_data_root",
        default="../MNIST/",
        type=str,
        help="Where to load mnist data",
        )
    parser.add_argument(
        "--mnist_determinstic",
        action="store_true",
        help="Whether to use determinstic moving mnist or not",
    )
    parser.add_argument(
        "--mnist_train_dataset_size",
        type=int,
        default=180000,
        help="Size of mnist training dataset"
    )
    parser.add_argument(
        "--mnist_val_dataset_size",
        type=int,
        default=20000,
        help="Size of mnist validation dataset"
    )
    parser.add_argument(
        "--mnist_linpred_dataset_size",
        type=int,
        default=10000,
        help="Size of mnist linpred dataset"
    )

    # Poke
    parser.add_argument(
        "--poke_data_root",
        type=str,
        default="../data/poke",
        help="Where to load carla data",
    )

    return parser

def build_dataset(args):
    if args.dataset == "dummy":
        train_dataset = DummyDataset(args.num_conditional_frames)
        val_dataset = DummyDataset(args.num_conditional_frames)

    if args.dataset == "moving_mnist":
        moving_mnist_transform = T.Normalize(
            mean=[0.1307,], std=[0.3081])
        train_dataset = MovingMNISTDataset(
            args.mnist_data_root,
            args.num_conditional_frames,
            moving_mnist_transform,
            train=True,
            height=64,
            width=64,
            ptp_type="2",
            num_digits=1,
            determinstic=args.mnist_determinstic,
            angle_range=(-30, 30),
            scale_range=(0.8, 1.2),
            shear_range=(-20, 20),
            dataset_size=args.mnist_train_dataset_size,
        )
        val_dataset = MovingMNISTDataset(
            args.mnist_data_root,
            args.num_conditional_frames,
            moving_mnist_transform,
            train=False,
            height=64,
            width=64,
            ptp_type="2",
            num_digits=1,
            determinstic=args.mnist_determinstic,
            angle_range=(-30, 30),
            scale_range=(0.8, 1.2),
            shear_range=(-20, 20),
            dataset_size=args.mnist_val_dataset_size
        )

        if args.no_linpred_eval:
            train_linpred_dataset = None
            val_linpred_dataset = None
        else:
            train_linpred_dataset = MovingMNISTDataset(
                args.mnist_data_root,
                1,
                moving_mnist_transform,
                train=True,
                dataset_size=args.mnist_linpred_dataset_size,
                linpred=True,
            )
            val_linpred_dataset = val_dataset

    if args.dataset == "poke":
        poke_transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
            T.Normalize([0.6501, 0.5889, 0.5466], [0.2822, 0.2744, 0.2606])
        ])
        train_dataset = PokeDataset(
            args.poke_data_root,
            args.num_conditional_frames,
            poke_transform,
            train=True,
        )
        val_dataset = PokeDataset(
            args.poke_data_root,
            args.num_conditional_frames,
            poke_transform,
            train=False,
        )
        train_linpred_dataset = PokeLinpredDataset(
            args.poke_data_root,
            poke_transform,
            train=True,
        )
        val_linpred_dataset = PokeLinpredDataset(
            args.poke_data_root,
            poke_transform,
            train=False,
        )

    return train_dataset, val_dataset, train_linpred_dataset, val_linpred_dataset
