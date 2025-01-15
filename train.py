import os
import csv
import glob
import torch
import argparse
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from scenario_tree_prediction import Encoder, Decoder
from torch.utils.data import DataLoader
from train_utils import *
from debug_utils import *


def train_epoch(data_loader, encoder, decoder, optimizer):
    epoch_loss = []
    epoch_metrics = []
    # train 方法是 nn.Module 提供的，它是一个内置的方法，用于将模型设置为训练模式。当你调用 .train() 方法时，PyTorch 会自动处理模型中所有层的状态，比如启用 Dropout 和 BatchNorm 层的训练特定行为。
    encoder.train()
    decoder.train()
    # data_loader：数据加载器，按批次（batch）加载训练数据。
    with tqdm(data_loader, desc="hailan-Training", unit="batch") as data_epoch:
        # 这是一个循环，用于遍历 data_epoch 迭代器中的每个批次。在每次迭代中，batch 将包含从 data_loader 加载的数据。
        # batch 通常包含 batch_size 个样本（data）

        # 都已经被dataloader转化为了张量
        for batch in data_epoch:
            # prepare data for predictor
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }
            # ground truth 也就是样本对应的标签数据
            # bt timesteps(80) xy heading
            ego_gt_future = batch[5].to(args.device)
            # bt neighbor timesteps(80) xy heading
            neighbors_gt_future = batch[6].to(args.device)
            print_tensor_shape_log(ego_gt_future, "ego_gt_future", DEBUG)
            print_tensor_shape_log(neighbors_gt_future, "neighbors_gt_future", DEBUG)
            # neighbors_future_valid：邻居车辆未来轨迹的有效性掩码（mask），用于过滤无效数据。
            # 这里使用省略号...来表示“所有前面的维度”，这意味着你将选择所有维度上的所有元素，直到最后一个维度。然后，:3表示你将选择最后一个维度上的前三个元素。这通常用于从每个样本中提取特定的特征。
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)
            print_tensor_shape_log(neighbors_future_valid, "neighbors_future_valid", DEBUG)
            # encode
            # NOTE 启用 PyTorch 的异常检测模式
            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad() # 清空优化器中存储的梯度，避免梯度累积。
            encoder_outputs = encoder(inputs) # 将输入数据传入编码器，生成编码器的输出（encoder_outputs）。



            # first stage prediction 0~3s的轨迹
            # batchsize branch time dim 
            first_stage_trajectory = batch[7].to(args.device)
            # 30 30个timesteps 0.1s一个时间步 第一阶段
            # neighbors_trajectories:bt branch neighbor times(0.1s切片) 3
            # scores： bt branch
            # ego：bt times 3
            # weights：bt 8
            
            # 输出：
            #     neighbors_trajectories：邻居车辆的预测轨迹。
            #     scores：预测轨迹的评分。
            #     ego：自车的预测轨迹。
            #     weights：预测的权重
            # 8s 80个timesteps
            neighbors_trajectories, scores, ego, weights = \
                decoder(encoder_outputs, first_stage_trajectory, inputs['neighbor_agents_past'], 30)
            print_tensor_shape_log(inputs['neighbor_agents_past'], "inputs['neighbor_agents_past']", DEBUG)
            print_tensor_shape_log(neighbors_trajectories, "neighbors_trajectories", DEBUG)
            print_tensor_shape_log(scores, "scores", DEBUG)
            print_tensor_shape_log(ego, "ego", DEBUG)
            print_tensor_shape_log(weights, "weights", DEBUG)
            loss = calc_loss(neighbors_trajectories, first_stage_trajectory, ego, scores, weights, \
                             ego_gt_future, neighbors_gt_future, neighbors_future_valid)
            # second stage prediction 0~8s的轨迹
            second_stage_trajectory = batch[8].to(args.device)
            print_tensor_shape_log(second_stage_trajectory, "second_stage_trajectory", DEBUG)
            # 输出：
            #     neighbors_trajectories：邻居车辆的预测轨迹。
            #     scores：预测轨迹的评分。
            #     ego：自车的预测轨迹。
            #     weights：预测的权重
            # 8s 80个timesteps
            neighbors_trajectories, scores, ego, weights = \
                decoder(encoder_outputs, second_stage_trajectory, inputs['neighbor_agents_past'], 80)
            # 论文第五页写到，考虑到闭环规划，第一阶段更重要，第二阶段没那么重要 所以系数分别是1.0 0.2
            loss += 0.2 * calc_loss(neighbors_trajectories, second_stage_trajectory, ego, scores, weights, \
                              ego_gt_future, neighbors_gt_future, neighbors_future_valid)
            print_tensor_shape_log(first_stage_trajectory, "first_stage_trajectory", DEBUG)
            print_tensor_shape_log(second_stage_trajectory, "second_stage_trajectory", DEBUG)

            # loss backward
            # 计算损失对模型参数的梯度。
            loss.backward()
            # 对梯度进行裁剪，防止梯度爆炸。
            # 参数 5.0 是梯度裁剪的阈值。
            # 在PyTorch中，torch.nn.utils.clip_grad_norm_ 是一个实用工具，用于在训练过程中对梯度（gradients）进行裁剪（clipping），
            # 以避免梯度爆炸（gradient explosion）问题。梯度爆炸是指在训练深度学习模型时，梯度值变得非常大，导致权重更新过大，进而引起训练不稳定。
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            # 根据梯度更新模型参数。
            optimizer.step()




            # compute metrics
            metrics = calc_metrics(second_stage_trajectory, neighbors_trajectories, scores, \
                                   ego_gt_future, neighbors_gt_future, neighbors_future_valid)
            # ADE（Average Displacement Error）：预测轨迹与真实轨迹的平均误差。
            # FDE（Final Displacement Error）：预测轨迹终点与真实轨迹终点的误差。
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            # 进度条中显示当前的平均损失。
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    # planningADE 和 planningFDE：规划阶段的平均误差和终点误差。
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    # predictionADE 和 predictionFDE：预测阶段的平均误差和终点误差。
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [planningADE, planningFDE, predictionADE, predictionFDE]
    logging.info(f"plannerADE: {planningADE:.4f}, plannerFDE: {planningFDE:.4f}, " +
                 f"predictorADE: {predictionADE:.4f}, predictorFDE: {predictionFDE:.4f}\n")
        
    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, encoder, decoder):
    epoch_loss = []
    epoch_metrics = []
    # 用于将模型设置为评估模式。也就是推理预测模式的时候，会自动处理所有层的状态，比如启用Dropout和BatchNorm层的训练特定行为。
    encoder.eval()
    decoder.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
            # prepare data for predictor
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            ego_gt_future = batch[5].to(args.device)
            neighbors_gt_future = batch[6].to(args.device)
            # 这里使用省略号...来表示“所有前面的维度”，这意味着你将选择所有维度上的所有元素，直到最后一个维度。然后，:3表示你将选择最后一个维度上的前三个元素。这通常用于从每个样本中提取特定的特征。
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)

            # predict
            # 通过 torch.no_grad() 禁用梯度计算，节省内存和计算资源，因为预测阶段不需要反向传播。
            # 作用范围：with 语句块中的所有代码都不会计算梯度。
            with torch.no_grad():
                encoder_outputs = encoder(inputs)

                # first stage prediction
                first_stage_trajectory = batch[7].to(args.device)
                neighbors_trajectories, scores, ego, weights = \
                    decoder(encoder_outputs, first_stage_trajectory, inputs['neighbor_agents_past'], 30)
                loss = calc_loss(neighbors_trajectories, first_stage_trajectory, ego, scores, weights, \
                                 ego_gt_future, neighbors_gt_future, neighbors_future_valid)

                # second stage prediction
                second_stage_trajectory = batch[8].to(args.device)
                neighbors_trajectories, scores, ego, weights = \
                    decoder(encoder_outputs, second_stage_trajectory, inputs['neighbor_agents_past'], 80)
                loss += 0.2 * calc_loss(neighbors_trajectories, second_stage_trajectory, ego, scores, weights, \
                                  ego_gt_future, neighbors_gt_future, neighbors_future_valid)
 
            # compute metrics
            metrics = calc_metrics(second_stage_trajectory, neighbors_trajectories, scores, \
                                   ego_gt_future, neighbors_gt_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [planningADE, planningFDE, predictionADE, predictionFDE]
    logging.info(f"val-plannerADE: {planningADE:.4f}, val-plannerFDE: {planningFDE:.4f}, " +
                 f"val-predictorADE: {predictionADE:.4f}, val-predictorFDE: {predictionFDE:.4f}\n")

    return np.mean(epoch_loss), epoch_metrics



def model_training(args):
    # Logging
    log_path = f"./DTPP/training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    # initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))
    logging.info("------------- {} -------------".format(args.name))

    # set seed
    # 设置随机种子，确保训练过程的可重复性。固定随机数生成器的状态，使得每次运行时的数据加载、模型初始化等随机过程的结果一致。
    set_seed(args.seed)

    # set up model
    # TODO
    # .to()方法是PyTorch中用于将模型或张量移动到指定设备的函数。在深度学习中，"设备"可以是CPU或GPU，这决定了模型的参数和数据将存储在哪里以及计算将在哪里执行。
    # args.device是一个字符串，指定了目标设备。通常，这个字符串可以是"cpu"或"cuda"（如果系统中有NVIDIA GPU并且安装了CUDA）。
    encoder = Encoder().to(args.device)
    # encoder.parameters() 返回编码器模型的所有参数（即权重和偏置）。
    # for p in encoder.parameters() 遍历这些参数。
    # p.numel() 返回参数张量中的元素数量。
    # sum(...) 函数计算所有参数元素数量的总和。
    logging.info("Encoder Params: {}".format(sum(p.numel() for p in encoder.parameters())))
    # 解码器中使用的邻居数量。
    # 解码器中候选路径的最大数量。
    # 解码器中是否使用可变权重。
    decoder = Decoder(neighbors=args.num_neighbors, max_branch=args.num_candidates, \
                      variable_cost=args.variable_weights).to(args.device)
    logging.info("Decoder Params: {}".format(sum(p.numel() for p in decoder.parameters())))

    # set up optimizer
    # 使用 AdamW 优化器，它是 Adam 优化器的改进版本，加入了权重衰减（weight decay）以防止过拟合。
    # 模型要优化的参数
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
    # 使用 StepLR 调度器，每隔 step_size=5 个 epoch，将学习率乘以 gamma=0.5，实现学习率的逐步衰减。
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = DrivingData(glob.glob(os.path.join(args.train_set, '*.npz')), args.num_neighbors, args.num_candidates)
    valid_set = DrivingData(glob.glob(os.path.join(args.valid_set, '*.npz')), args.num_neighbors, args.num_candidates)
    # NOTE DataLoader 会将 NumPy 数组自动转换为 PyTorch 张量!!!
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=os.cpu_count())
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=os.cpu_count())
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    
    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        train_loss, train_metrics = train_epoch(train_loader, encoder, decoder, optimizer)
        val_loss, val_metrics = valid_epoch(valid_loader, encoder, decoder)

        # save to training log
        # NOTE 训练过程中的loss和指标信息
        log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
               'train-planningADE': train_metrics[0], 'train-planningFDE': train_metrics[1], 
               'train-predictionADE': train_metrics[2], 'train-predictionFDE': train_metrics[3],
               'val-planningADE': val_metrics[0], 'val-planningFDE': val_metrics[1], 
               'val-predictionADE': val_metrics[2], 'val-predictionFDE': val_metrics[3]}

        if epoch == 0:
            # 如果文件不存在，将会创建它；如果文件已存在，它的内容将被清空。
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # reduce learning rate
        # 在每个 epoch 结束后，调整优化器的学习率。
        scheduler.step()

        # save model at the end of epoch
        # 提取编码器（encoder）的所有参数，返回一个字典，键是参数的名称，值是参数的张量（Tensor）。
        model = {'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}
        torch.save(model, f'training_log/{args.name}/model_epoch_{epoch+1}_valADE_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name', default="DTPP_training")
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--train_set', type=str, help='path to training data')
    parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=10)
    # NOTE 自车候选轨迹的数量
    parser.add_argument('--num_candidates', type=int, help='number of max candidate trajectories', default=30)
    parser.add_argument('--variable_weights', type=bool, help='use variable cost weights', default=False)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=30)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=1e-5)
    parser.add_argument('--device', type=str, help='run on which device', default='cuda')
    args = parser.parse_args()

    # Run model training
    model_training(args)