import os
import pickle
import time
import warnings
import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
from tqdm import tqdm

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.ddp:
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            self.device = self.args.gpu

        model = self.model_dict[self.args.model].Model(self.args)

        if self.args.adaptation:
            model.load_state_dict(torch.load(self.args.pretrain_model_path))

        if self.args.ddp:
            model = DDP(model.cuda(), device_ids=[self.args.local_rank])
        elif self.args.dp:  # 数据并行
            model = DataParallel(model, device_ids=self.args.device_ids).to(self.device)
        else:
            self.device = self.args.gpu
            model = model.to(self.device)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        p_list = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            else:
                p_list.append(p)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, is_test=False):

        total_loss = []
        total_count = []
        iter_count = 0

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_date) in tqdm(enumerate(vali_loader),
                                                                                      total=len(vali_loader),
                                                                                      desc=f'VALID {iter_count}'):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_date = batch_date.int().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_date, batch_x_mark, batch_y_mark)

                if is_test or self.args.nonautoregressive:
                    outputs = outputs[:, -self.args.output_token_len:, :]
                    batch_y = batch_y[:, -self.args.output_token_len:, :].to(self.device)
                else:
                    outputs = outputs[:, :, :]
                    batch_y = batch_y[:, :, :].to(self.device)

                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                loss = criterion(outputs, batch_y)

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
        if self.args.ddp:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.reduce(total_loss, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)

        self.model.train()

        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
            if not os.path.exists(path):
                os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(self.args, verbose=True)
        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion()
        print(f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_date) in tqdm(enumerate(train_loader),
                                                                                      total=len(train_loader),
                                                                                      desc=f"TRAIN epoch {epoch}"):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_date = batch_date.int().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_date, batch_x_mark, batch_y_mark)

                if self.args.dp:
                    torch.cuda.synchronize()
                if self.args.nonautoregressive:
                    batch_y = batch_y[:, -self.args.output_token_len:, :]
                if self.args.covariate:
                    if self.args.last_token:
                        outputs = outputs[:, -self.args.output_token_len:, -1]
                        batch_y = batch_y[:, -self.args.output_token_len:, -1]
                    else:
                        outputs = outputs[:, :, -1]
                        batch_y = batch_y[:, :, -1]
                batch_y = batch_y[:, -outputs.shape[1]:, :]

                loss = criterion(outputs, batch_y)
                loss.backward()
                model_optim.step()

            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

            vali_loss = self.vali(vali_data, vali_loader, criterion, is_test=self.args.valid_last)
            test_loss = self.vali(test_data, test_loader, criterion, is_test=True)
            if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                print("Epoch: {}, Steps: {} | Vali Loss: {:.7f} Test Loss: {:.7f}".format(
                    epoch + 1, train_steps, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)

            # 强行保存每个checkpoints
            if self.args.dp:
                save_model = self.model.module
            else:
                save_model = self.model
            param_grad_dic = {
                k: v.requires_grad for (k, v) in save_model.named_parameters()
            }
            state_dict = save_model.state_dict()
            for k in list(state_dict.keys()):
                if k in param_grad_dic.keys() and not param_grad_dic[k]:
                    del state_dict[k]
            torch.save(state_dict, path + '/' + f'checkpoint_{epoch}.pth')
            print(f'保存检查点{epoch}成功')

            if early_stopping.early_stop:
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                if (self.args.ddp and self.args.local_rank == 0) or not self.args.ddp:
                    print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)
            if self.args.ddp:
                train_loader.sampler.set_epoch(epoch + 1)
        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.ddp:
            dist.barrier()
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        else:
            self.model.load_state_dict(torch.load(best_model_path), strict=False)
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("info:", self.args.test_seq_len, self.args.input_token_len, self.args.output_token_len,
              self.args.test_pred_len)
        if test:
            print('loading model')
            setting = self.args.test_dir
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            checkpoint = torch.load(os.path.join(self.args.checkpoints, setting, best_model_path))
            for name, param in self.model.named_parameters():
                if not param.requires_grad and name not in checkpoint:
                    checkpoint[name] = param
            self.model.load_state_dict(checkpoint)

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        iter_count = 0
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_date) in tqdm(enumerate(test_loader),
                                                                                      total=len(test_loader),
                                                                                      desc=f'TEST {iter_count}'):
                iter_count += 1
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_date = batch_date.int().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                inference_steps = self.args.test_pred_len // self.args.output_token_len
                dis = self.args.test_pred_len - inference_steps * self.args.output_token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []
                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.input_token_len:, :], pred_y[-1]], dim=1)

                    outputs = self.model(batch_x, batch_date, batch_x_mark, batch_y_mark)

                    pred_y.append(outputs[:, -self.args.output_token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-self.args.output_token_len + dis, :]
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                outputs = pred_y.detach().cpu()
                batch_y = batch_y.detach().cpu()
                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        if self.args.covariate:
            preds = preds[:, :, -1]
            trues = trues[:, :, -1]
        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)
        print(f'mse:{mse:.5f}, mae:{mae:.5f}')
        return
