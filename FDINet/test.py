import torch
from dataset.dataset import MyDataset
import tqdm
import cv2
from torch.utils.data import DataLoader
from metrics.metric_tool import ConfuseMatrixMeter

from models.change_classifier import Demo as Model
import argparse

def parse_arguments():
    # Argument Parser creation
    parser = argparse.ArgumentParser(
        description="Parameter for data analysis, data cleaning and model training."
    )
    parser.add_argument(
        "--datapath",
        type=str,
        help="data path",
        default="/home/tang/TANG/Dataset/ChangeDetection/WH/data_256",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        help="model path",
        default="/home/tang/TANG/Code/demo/test_net/Demo/wh/ce/model_125.pth",
    )
    parser.add_argument(
        "--result_save_path",
        type=str,
        help="result path",
        default="/home/tang/TANG/Code/demo/vis/FINet/"
    )

    parsed_arguments = parser.parse_args()

    return parsed_arguments


def swap_mask(mask):
    swap = torch.where(mask == 1, torch.tensor(2), torch.where(mask == 2, torch.tensor(1), mask))
    return swap


def evaluate(x1, x2, mask, tool4metric):
    # All the tensors on the device:
    x1 = x1.to(device).float()
    x2 = x2.to(device).float()

    # imageA = x1.unsqueeze(2)
    # imageB = x2.unsqueeze(2)
    # images = torch.cat([imageB, imageA], 2)
    # mask = swap_mask(mask.long())
    mask = mask.to(device).long()

    # Evaluating the model:
    # pred = model(x1, x2)
    pred, _ = model(x1, x2)

    # Loss gradient descend step:

    # Feeding the comparison metric tool:
    # bin_genmask = (pred.to("cpu") > 0.5).detach().numpy().astype(int)
    # mask = mask.to("cpu").numpy().astype(int)
    # tool4metric.update_cm(pr=bin_genmask, gt=mask)

    _, pred = torch.max(pred.data, dim=1)
    tool4metric.update_cm(pr=pred.to("cpu").numpy(), gt=mask.to("cpu").numpy())

    return pred.squeeze()


if __name__ == "__main__":

    # Parse arguments:
    args = parse_arguments()

    # Initialisation of the dataset
    data_path = args.datapath
    dataset = MyDataset(data_path, "test")
    test_loader = DataLoader(dataset, batch_size=1)

    # Initialisation of the model and print model stat
    model = Model(3, 3)
    modelpath = args.modelpath
    model.load_state_dict(torch.load(modelpath))

    tool4metric = ConfuseMatrixMeter(n_class=3)

    # Set evaluation mode and cast the model to the desidered device
    model.eval()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model.to(device)

    # loop to evaluate the model and print the metrics
    tool4metric.clear()
    with torch.no_grad():
        for (reference, testimg), mask, name in tqdm.tqdm(test_loader):
            pred = evaluate(reference, testimg, mask, tool4metric)

            cv2.imwrite(args.result_save_path + "/WH/ce/label/" + ''.join(name) + '.png', pred.cpu().numpy())

    scores_dictionary = tool4metric.get_scores()
    epoch_result = 'mF1_score = {}, mIoU = {}, Acc = {}\n iou_0 = {}, iou_1 = {}, iou_2 = {}'.format(
        scores_dictionary['mf1'],
        scores_dictionary['miou'],
        scores_dictionary['acc'],
        scores_dictionary['iou_0'],
        scores_dictionary['iou_1'],
        scores_dictionary['iou_2'])
    print(epoch_result)
    print()










