import itertools
from src import utils as ut

quick_opt_list = [
    {'name': 'adam'},
    {
        "name": "chyps",
        "init_step_size": 1e-3,  # Bạn có thể tinh chỉnh learning rate khởi tạo ở đây
        "gamma": 0.9,
        "epsilon": 1e-8,
        "beta": 10.0,
        "tau": 2.0,
        "option": "II"
    }
]

EXP_GROUPS = {}

EXP_GROUPS['quick_cifar10_test'] = ut.cartesian_exp_group({
    "dataset": ["cifar10"],
    "model": ["resnet34"],             # Sử dụng ResNet-34 chuẩn
    "loss_func": ["softmax_loss"],     # Dùng Cross Entropy cho Multi-class
    "acc_func": ["softmax_accuracy"],
    "opt": quick_opt_list,
    "batch_size": [128],               # Batch size chuẩn cho CIFAR
    "max_epoch": [10],                 # Chạy đúng 10 epoch để check code
    "runs": [0]
})