import argparse
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F


import dataset.dataset as dtset
import torch
import numpy as np
import random
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as autocast
from metrics.metric_tool import ConfuseMatrixMeter
from models.change_classifier import Demo as Model
# from models.Models import SNUNet_ECAM as Model
# from models.LSNet.Models import LSNet_denseFPN as Model
# from models.AFCF3D.Networks import Model as Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
import loss_F


def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        default="/home/tang/TANG/Dataset/ChangeDetection/WH/data_256",
        type=str,
        help="data path",
    )
    #/home/tang/TANG/Dataset/ChangeDetection/SECOND/data
    #/home/tang/TANG/Dataset/ChangeDetection/WH/data_256
    #/home/tang/TANG/Dataset/ChangeDetection/CDD_fine/data
    parser.add_argument(
        "--log-path",
        default="/home/tang/TANG/Code/Object_Fine-Grained_CD/demo/test_net/",
        type=str,
        help="log path",
    )

    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')

    parsed_arguments = parser.parse_args()

    # create log dir if it doesn't exists
    if not os.path.exists(parsed_arguments.log_path):
        os.mkdir(parsed_arguments.log_path)

    dir_run = sorted(
        [
            filename
            for filename in os.listdir(parsed_arguments.log_path)
            if filename.startswith("run_")
        ]
    )

    if len(dir_run) > 0:
        num_run = int(dir_run[-1].split("_")[-1]) + 1
    else:
        num_run = 0
    parsed_arguments.log_path = os.path.join(
        parsed_arguments.log_path, "run_%04d" % num_run + "/"
    )

    return parsed_arguments

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.

    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)
    return result


def swap_mask(mask):
    swap = torch.where(mask == 1, torch.tensor(2), torch.where(mask == 2, torch.tensor(1), mask))
    return swap


def train(
    dataset_train,
    dataset_val,
    scaler,
    model,
    optimizer,
    scheduler,
    logpath,
    writer,
    epochs,
    device
):

    model = model.to(device)

    tool4metric = ConfuseMatrixMeter(n_class=3)

    def training_phase(epc):
        print("Epoch {}".format(epc))
        model.train()
        epoch_loss = 0.0
        loop = tqdm(dataset_train, file=sys.stdout)
        for x1, x2, mask, name in loop:
            optimizer.zero_grad()

            x1 = x1.to(device).float()
            x2 = x2.to(device).float()

            # imageA = x1.unsqueeze(2)
            # imageB = x2.unsqueeze(2)
            # images = torch.cat([imageA, imageB], 2)

            mask = mask.to(device).long()

            with autocast():
                # Evaluating the model:
                pred1, pred2 = model(x1, x2)

                label = make_one_hot(mask.unsqueeze(1), 3).squeeze(0).to(device)
                cd_loss = loss_F.change_dice_loss(pred1, pred2)
                # cd_loss = loss_F.dice_loss(pred1, label)
                ce_loss = nn.CrossEntropyLoss()(pred1, label)
                total_loss = ce_loss + cd_loss

            # Reset the gradients:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track metrics:
            epoch_loss += total_loss.to("cpu").detach().numpy()
            ### end of iteration for epoch ###

        epoch_loss /= len(dataset_train)

        #########
        print("Training phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss))
        writer.add_scalar("Loss/epoch", epoch_loss, epc)
        writer.flush()

    def validation_phase(epc):
        model.eval()
        epoch_loss_eval = 0.0
        tool4metric.clear()
        loop = tqdm(dataset_val, file=sys.stdout)
        with torch.no_grad():
            for x1, x2, mask, name in loop:

                x1 = x1.to(device).float()
                x2 = x2.to(device).float()

                # imageA = x1.unsqueeze(2)
                # imageB = x2.unsqueeze(2)
                # images = torch.cat([imageA, imageB], 2)
                # swap = swap_mask(mask.long())
                # mask = mask.to(device).long()
                # mask2 = swap.to(device).long()
                mask = mask.to(device).long()

                # Evaluating the model:
                with autocast():
                    pred1, pred2 = model(x1, x2)

                    label = make_one_hot(mask.unsqueeze(1), 3).squeeze(0).to(device)
                    cd_loss = loss_F.change_dice_loss(pred1, pred2)
                    # cd_loss = loss_F.dice_loss(pred1, label)
                    ce_loss = nn.CrossEntropyLoss()(pred1, label)
                    total_loss = ce_loss + cd_loss

                epoch_loss_eval += total_loss

                # Feeding the comparison metric tool:
                # pred = F.softmax(pred1, dim=1)
                _, pred = torch.max(pred1.data, dim=1)
                tool4metric.update_cm(pr=pred.to("cpu").numpy(), gt=mask.to("cpu").numpy())

        epoch_loss_eval /= len(dataset_val)
        print("Validation phase summary")
        print("Loss for epoch {} is {}".format(epc, epoch_loss_eval))
        writer.add_scalar("Loss_val/epoch", epoch_loss_eval, epc)
        scores_dictionary = tool4metric.get_scores()
        epoch_result = 'mF1_score = {}, mIoU = {}'.format(
            scores_dictionary['mf1'],
            scores_dictionary['miou'])
        print(epoch_result)
        print()

        return scores_dictionary['miou']

    score = 0

    for epc in range(epochs):
        training_phase(epc)
        f1 = validation_phase(epc)

        if f1 > score:
            score = f1
            torch.save(model.state_dict(), os.path.join(logpath, "model_{}.pth".format(epc)))

        # scheduler step
        scheduler.step()


def run():

    # set the random seed
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Parse arguments:
    args = parse_arguments()

    # Initialize tensorboard:
    writer = SummaryWriter(log_dir=args.log_path)

    # Inizialitazion of dataset and dataloader:
    trainingdata = dtset.MyDataset(args.datapath, "train")
    validationdata = dtset.MyDataset(args.datapath, "val")
    data_loader_training = DataLoader(trainingdata, batch_size=32, shuffle=True)
    data_loader_val = DataLoader(validationdata, batch_size=8, shuffle=True)

    # device setting for training
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')

    print(f'Current Device: {device}\n')

    # Initialize the model
    model = Model(3, 3)
    restart_from_checkpoint = False
    model_path = None
    if restart_from_checkpoint:
        model.load_state_dict(torch.load(model_path))
        print("Checkpoint succesfully loaded")

    # print number of parameters
    parameters_tot = 0
    for nom, param in model.named_parameters():
        # print (nom, param.data.shape)
        parameters_tot += torch.prod(torch.tensor(param.data.shape))
    print("Number of model parameters {}\n".format(parameters_tot))

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001, amsgrad=False)
    # # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    # copy the configurations
    _ = shutil.copytree(
        "./models",
        os.path.join(args.log_path, "models"),
    )

    train(
        data_loader_training,
        data_loader_val,
        scaler,
        model,
        optimizer,
        scheduler,
        args.log_path,
        writer,
        epochs=200,
        device=device
    )
    writer.close()


if __name__ == "__main__":
    run()
