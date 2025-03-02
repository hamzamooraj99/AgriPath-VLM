'''
# summary_writer.py
## Author: @hamzamooraj99 (Hamza Hassan Mooraj)
Description: This file contains a script to write a summary of all CNN model experiments using TensorBoard
'''

import torch
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
import resnet50_lightning as rn
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC, AveragePrecision, MatthewsCorrCoef, CohenKappa
import matplotlib.pyplot as plt
import numpy as np
import os

#region CONFIGURATION
EXPERIMENTS = {
    "ResNet50 Batch=16 LR=1e-4": (r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\resnet50_agripath_exp_0.pth", 16, 1e-4),
    "ResNet50 Batch=16 LR=5e-4": (r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\resnet50_agripath_exp_1.pth", 16, 5e-4),
    "ResNet50 Batch=16 LR=2e-4": (r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\resnet50_agripath_exp_2.pth", 16, 2e-4),
    "ResNet50 Batch=32 LR=1e-4": (r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\resnet50_agripath_exp_3.pth", 32, 1e-4),
    "ResNet50 Batch=32 LR=5e-4": (r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\resnet50_agripath_exp_4.pth", 32, 5e-4),
    "ResNet50 Batch=32 LR=2e-4": (r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\resnet50_agripath_exp_5.pth", 32, 2e-4),
    "ResNet50 Batch=64 LR=1e-4": (r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\resnet50_agripath_exp_6.pth", 64, 1e-4),
    "ResNet50 Batch=64 LR=5e-4": (r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\resnet50_agripath_exp_7.pth", 64, 5e-4),
    "ResNet50 Batch=64 LR=2e-4": (r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\resnet50_agripath_exp_8.pth", 64, 2e-4)
}

HF_REPO = "hamzamooraj99/AgriPath-CNN"
NUM_CLASSES = 65
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOGGER_DIR = r"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\tb_logs"

os.makedirs(LOGGER_DIR, exist_ok=True)

DataModule = rn.AgriPathDataModule
ModelModule = rn.ResNet50TLModel
#endregion

def plot_confusion_matrix(conf_mat):
    
    conf_mat = conf_mat.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(conf_mat, cmap=plt.cm.Blues)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return fig

def evaluate_model(exp_name, path, batch_size, learning_rate, num_classes = NUM_CLASSES):
    print(f"\nEvaluating {exp_name}...")

    model = ModelModule(num_classes=num_classes, learning_rate=learning_rate)
    checkpoint = torch.load(path, map_location=torch.device(DEVICE), weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    datamodule = DataModule(HF_REPO, batch_size=batch_size)
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()

    acc = Accuracy(task='multiclass', num_classes=num_classes).to(DEVICE)
    pr = Precision(task='multiclass', num_classes=num_classes, average='macro').to(DEVICE)
    re = Recall(task='multiclass', num_classes=num_classes, average='macro').to(DEVICE)
    f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(DEVICE)
    cm = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(DEVICE)
    auroc = AUROC(task='multiclass', num_classes=num_classes, average='macro').to(DEVICE)
    mcc = MatthewsCorrCoef(task='multiclass', num_classes=num_classes).to(DEVICE)
    kappa = CohenKappa(task='multiclass', num_classes=num_classes).to(DEVICE)
    bal_acc = Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(DEVICE)
    pr_pClass = Precision(task='multiclass', num_classes=num_classes, average=None).to(DEVICE)
    re_pClass = Recall(task='multiclass', num_classes=num_classes, average=None).to(DEVICE)
    f1_pClass = F1Score(task='multiclass', num_classes=num_classes, average=None).to(DEVICE)

    batch_iter = 0
    for batch in test_loader:
        # print(f"Batch iter {batch_iter}")
        x, y = batch
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = model(x)
        preds = torch.argmax(y_hat, dim=1)

        acc.update(preds, y)
        pr.update(preds, y)
        re.update(preds, y)
        f1.update(preds, y)
        cm.update(preds, y)
        auroc.update(y_hat, y)
        mcc.update(preds, y)
        kappa.update(preds, y)
        bal_acc.update(preds, y)
        pr_pClass.update(preds, y)
        re_pClass.update(preds, y)
        f1_pClass.update(preds, y)

        batch_iter+=1
    
    accuracy = acc.compute()
    precision = pr.compute()
    recall = re.compute()
    f1_score = f1.compute()
    conf_matrix = cm.compute()
    auroc_score = auroc.compute()
    mcc_score = mcc.compute()
    cohen_kappa = kappa.compute()
    balanced_accuracy = bal_acc.compute()
    precision_per_class = pr_pClass.compute()
    recall_per_class = re_pClass.compute()
    f1_per_class = f1_pClass.compute()

    print("Logging...")

    logger = TensorBoardLogger(LOGGER_DIR, name=exp_name)
    logger.experiment.add_scalar("Accuracy", accuracy, global_step=0)
    logger.experiment.add_scalar("Precision", precision, global_step=0)
    logger.experiment.add_scalar("Recall", recall, global_step=0)
    logger.experiment.add_scalar("F1 Score", f1_score, global_step=0)
    logger.experiment.add_scalar("AUROC", auroc_score, global_step=0)
    logger.experiment.add_scalar("MCC", mcc_score, global_step=0)
    logger.experiment.add_scalar("Cohen Kappa", cohen_kappa, global_step=0)
    logger.experiment.add_scalar("Balanced Accuracy", balanced_accuracy, global_step=0)

    for i in range(num_classes):
        logger.experiment.add_scalar(f"Precision/Class{i}", precision_per_class[i], global_step=0)
        logger.experiment.add_scalar(f"Recall/Class{i}", recall_per_class[i], global_step=0)
        logger.experiment.add_scalar(f"F1 Score/Class{i}", f1_per_class[i], global_step=0)

    fig = plot_confusion_matrix(conf_matrix)
    logger.experiment.add_figure("Confusion Matrix", fig, global_step=0)
    plt.close(fig)

    log_filename = (fr"C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\experiment_logs\{exp_name}.txt")

    with open(log_filename, 'w') as log_file:
        log_file.write(f"Batch Size: {batch_size}, Learning Rate: {learning_rate}\n")
        log_file.write(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1_score}\n")
        # log_file.write(f"Confusion Matrix:\n {conf_matrix}")
        log_file.write(f"Balanced Accuracy: {balanced_accuracy}\n")
        log_file.write(f"Matthews Correlation Coefficient (MCC): {mcc_score}\n")
        log_file.write(f"Cohen’s Kappa: {cohen_kappa}\n")
        log_file.write("Per-Class Precision, Recall, and F1 Scores:\n")
        for i in range(num_classes):
            log_file.write(f"Class {i} - Precision: {precision_per_class[i]}, Recall: {recall_per_class[i]}, F1: {f1_per_class[i]}\n")
        log_file.write("\n" + "-"*50 + "\n")
    print(f"Logged results in file: {exp_name}.txt\n")


if __name__ == '__main__':
    for exp, details in EXPERIMENTS.items():
        path, batch, lr = details[0], details[1], details[2]
        evaluate_model(exp_name=exp, path=path, batch_size=batch, learning_rate=lr)

    print(f"Evaluation Complete. Run `tensorboard --logdir={LOGGER_DIR}` to view results.")

#tensorboard --logdir="C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\tb_logs" --outdir="C:\Users\Hamza\Documents\Heriot Watt\HWU-Y4\Dissertation\D2\Models\CNN\resnet50\tb_data"