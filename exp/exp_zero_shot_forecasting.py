from data_provider.data_factory import data_provider
from data_provider.m4 import M4Meta
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.losses import zero_shot_smape_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.utils.data import Subset
from numpy.random import choice
from omegaconf import OmegaConf
from tqdm import tqdm
import sys
from utils.metrics import metric
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def SMAPE(pred, true):
    return np.mean(200 * np.abs(pred - true) / (np.abs(pred) + np.abs(true) + 1e-8))
def MAPE(pred, true):
    return np.mean(np.abs(100 * (pred - true) / (true +1e-8)))
def get_init_config(config_path=None):
    config = OmegaConf.load(config_path)
    return config
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        # f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        f"trainable params: {trainable_params} || all params: {all_param}"
    )
class Exp_Zero_Shot_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Zero_Shot_Forecast, self).__init__(args)

    def _build_model(self):
        if self.args.data == 'tsf':
            self.args.frequency_map = M4Meta.frequency_map[self.args.seasonal_patterns]
        self.device = self.args.gpu
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
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
                print(n, p.dtype, p.shape)
        model_optim = optim.Adam([{'params': p_list}], lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        print('next learning rate is {}'.format(self.args.learning_rate))
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'SMAPE':
            return zero_shot_smape_loss()
        elif loss_name =="MAE":
            return nn.L1Loss()

    def _update_args_from_config(self, args, config, dataset_name):
        """Update args with dataset specific configurations"""
        dataset_config = config['datasets'][dataset_name]
        for key in ['data', 'root_path', 'data_path', 'data_name', 'features',
                    'freq', 'target', 'embed', 'percent', 'lradj']:
            setattr(args, key, getattr(dataset_config, key))
        
        if args.freq == 0:
            args.freq = 'h'

    def _combine_datasets(self, datasets):
        """Combine multiple datasets into one"""
        combined = datasets[0]
        for dataset in datasets[1:]:
            combined = torch.utils.data.ConcatDataset([combined, dataset])
        return combined
    def print_dataset_info(self, data, loader, name="Dataset"):
        print(f"\n=== {name} Information ===")
        print(f"Number of samples: {len(data)}")
        print(f"Batch size: {loader.batch_size}")
        print(f"Number of batches: {len(loader)}")
        
        
        for attr in ['features', 'targets', 'shape']:
            if hasattr(data, attr):
                print(f"{attr}: {getattr(data, attr)}")

    def train(self, setting):
        # train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')

        # self.args.data_path = self.args.test_data_path
        # test_data2, test_loader2 = self._get_data(flag='test')
        
        config = get_init_config(self.args.config_path)

        train_data_name = self.args.datasets.split(',')
        print(train_data_name)
        train_datas = []
        val_datas = []
        min_sample_num = sys.maxsize

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # First pass to get validation data and minimum sample number
        for dataset_name in self.args.datasets.split(','):
            self._update_args_from_config(self.args, config, dataset_name)
            
            train_data, train_loader = data_provider(self.args, 'train')
            if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange']:
                min_sample_num = min(min_sample_num, len(train_data))
        
        for dataset_name in self.args.eval_data.split(','):  
            self._update_args_from_config(self.args, config, dataset_name)  
            val_data, val_loader = data_provider(self.args, 'val') 
            val_datas.append(val_data)

        # Second pass to prepare training data with proper sampling
        for dataset_name in self.args.datasets.split(','):
            self._update_args_from_config(self.args, config, dataset_name)
            
            train_data, _ = data_provider(self.args, 'train')
            
            if dataset_name not in ['ETTh1', 'ETTh2', 'ILI', 'exchange'] and self.args.equal == 1:
                train_data = Subset(train_data, choice(len(train_data), min_sample_num))
                
            if self.args.equal == 1:
                if dataset_name == 'electricity' and self.args.electri_multiplier > 1:
                    train_data = Subset(train_data, choice(len(train_data), 
                                    int(min_sample_num * self.args.electri_multiplier)))
                elif dataset_name == 'traffic' and self.args.traffic_multiplier > 1:
                    train_data = Subset(train_data, choice(len(train_data),
                                    int(min_sample_num * self.args.traffic_multiplier)))
                    
            train_datas.append(train_data)
        
        if len(train_datas) > 1:
            train_data = self._combine_datasets(train_datas)
            val_data = self._combine_datasets(val_datas)
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size, 
                                    shuffle=True, num_workers=self.args.num_workers)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=self.args.batch_size,
                                shuffle=False, num_workers=self.args.num_workers)

        # Prepare test data
        self._update_args_from_config(self.args, config, self.args.target_data)
        print(self.args)
        test_data, test_loader = data_provider(self.args, 'test')

        self.print_dataset_info(train_data, train_loader, "Training Dataset")
        self.print_dataset_info(val_data, val_loader, "Validation Dataset")
        self.print_dataset_info(test_data, test_loader, "Test Dataset")            

        time_now = time.time()

        train_steps = len(train_loader)
        print("Train steps:",train_steps)
        print_trainable_parameters(self.model)
        early_stopping = EarlyStopping(self.args, verbose=True)

        model_optim = self._select_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.tmax, eta_min=1e-8)
        criterion = self._select_criterion(self.args.loss)
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in tqdm(enumerate(train_loader),total = len(train_loader)):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                seq_trend = seq_trend.float().to(self.device)
                seq_seasonal = seq_seasonal.float().to(self.device)
                seq_resid = seq_resid.float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, orthogonal_loss, simlarity_loss = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                        # outputs = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                else:
                    outputs, orthogonal_loss, simlarity_loss = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                    # outputs = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                # orthogonal_loss, simlarity_loss = self.model.orthogonal_loss()
                loss = criterion(outputs, batch_y)
                # loss = loss
                loss = loss + 0.00001 * orthogonal_loss
                # loss = loss - 0.05 * simlarity_loss

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward(retain_graph=True)
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward(retain_graph=True)
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print("Epoch: {} average loss: {}".format(epoch + 1, np.average(train_loss)))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(val_data, val_loader, criterion)  
            test_loss = self.vali2(test_data, test_loader, criterion)    # test_loss indicates the result on the source datasets, 
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            if self.args.cosine:
                scheduler.step()
                print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + f'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path), strict=False)

        return self.model

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float()
                seq_trend = seq_trend.float().to(self.device)
                seq_seasonal = seq_seasonal.float().to(self.device)
                seq_resid = seq_resid.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, orthogonal_loss, simlarity_loss = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                        # outputs = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                else:
                    outputs, orthogonal_loss, simlarity_loss = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                    # outputs = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        
        return total_loss

    def vali2(self, vali_data, vali_loader, criterion):
        total_loss = []
        count= []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float()
                seq_trend = seq_trend.float().to(self.device)
                seq_seasonal = seq_seasonal.float().to(self.device)
                seq_resid = seq_resid.float().to(self.device)
                
                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []

                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, orthogonal_loss, simlarity_loss = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                            # outputs = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                    else:
                        outputs, orthogonal_loss, simlarity_loss = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                        # outputs = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                    pred_y.append(outputs[:, -self.args.token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-dis, :]
                
                outputs = pred_y
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)
                count.append(batch_x.shape[0])
        total_loss = np.average(total_loss, weights=count)
        self.model.train()
        
        return total_loss    

    def test(self, setting, test=0):
        if test:
            print('loading model')
            # setting = self.args.test_dir
            print(setting)
            best_model_path = self.args.test_file_name
            print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, best_model_path)), strict=False)
            # print_trainable_parameters(self.model)
        config = get_init_config(self.args.config_path)
        # Prepare test data
        self._update_args_from_config(self.args, config, self.args.target_data)
        self.args.data_path = self.args.test_data_path
        test_data, test_loader = self._get_data('test')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, seq_trend, seq_seasonal, seq_resid) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y = batch_y.float()
                seq_trend = seq_trend.float().to(self.device)
                seq_seasonal = seq_seasonal.float().to(self.device)
                seq_resid = seq_resid.float().to(self.device)

                inference_steps = self.args.test_pred_len // self.args.token_len
                dis = self.args.test_pred_len - inference_steps * self.args.token_len
                if dis != 0:
                    inference_steps += 1
                pred_y = []

                for j in range(inference_steps):
                    if len(pred_y) != 0:
                        batch_x = torch.cat([batch_x[:, self.args.token_len:, :], pred_y[-1]], dim=1)
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            outputs, orthogonal_loss, simlarity_loss = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                            # outputs = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                    else:
                        outputs, orthogonal_loss, simlarity_loss = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                        # outputs = self.model(batch_x, batch_x_mark, None, None, seq_trend, seq_seasonal, seq_resid)
                    pred_y.append(outputs[:, -self.args.token_len:, :])
                pred_y = torch.cat(pred_y, dim=1)
                if dis != 0:
                    pred_y = pred_y[:, :-dis, :]
                
                outputs = pred_y
                batch_y = batch_y[:, -self.args.test_pred_len:, :].to(self.device)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)
                
                if self.args.visualize and i == 0:
                    gt = np.array(true[0, :, -1])
                    pd = np.array(pred[0, :, -1])
                    lookback = batch_x[0, :, -1].detach().cpu().numpy()
                    gt = np.concatenate([lookback, gt], axis=0)
                    pd = np.concatenate([lookback, pd], axis=0)
                    dir_path = folder_path + f'{self.args.test_pred_len}/'
                    if not os.path.exists(dir_path):
                        os.makedirs(dir_path)
                    visual(self.args, gt, pd, os.path.join(dir_path, f'{self.args.test_pred_len}_{i}.png'))
                
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()

    def predict(self):
        """
        Inference + token decoding + visualization of time series forecast with decoded GPT-2 tokens.
        """
        print('loading model')
        # setting = self.args.test_dir
        # best_model_path = self.args.test_file_name
        # print("loading model from {}".format(os.path.join(self.args.checkpoints, setting, best_model_path)))
        self.model.load_state_dict(torch.load("/mnt/HDD1/thanglq/AutoTimes/checkpoints/zero_shot_forecast_ETTh1_672_96_1024_no_pca_AutoTimes_Gpt2_ETTh1_sl672_ll576_tl96_lr0.001_bt1024_wd0_hd2048_hl2_cosTrue_mixFalse_test_0/checkpoint.pth"), strict=False)
        self.model.eval()
        self.args.data_path = self.args.test_data_path
        test_data, test_loader = self._get_data('test')
        all_forecasts = []
        all_tokens = []

        tokenizer = self.model.tokenizer  # already initialized in the model
        save_path = os.path.join('./visual_predictions', self.args.model_id)
        os.makedirs(save_path, exist_ok=True)

        with torch.no_grad():
            for i, (batch_x, _, batch_x_mark, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)

                forecast_output, gpt2_outputs = self.model.forecast_return_gpt2(
                    batch_x, batch_x_mark, None, None
                )

                logits = self.model.lm_head(gpt2_outputs)
                token_ids = logits.argmax(dim=-1)
                decoded = [
                    [tokenizer.decode([token_id.item()], skip_special_tokens=True) for token_id in seq]
                    for seq in token_ids
                ]

                forecast_output = forecast_output.cpu().numpy()
                all_forecasts.append(forecast_output)
                all_tokens.extend(decoded)
                print(len(all_tokens), len(all_forecasts))
                # 🔍 Visualization (first few samples)
                for j in range(min(3, len(decoded))):
                    forecast = forecast_output[j, :, -1]  # visualize last variable
                    tokens = decoded[j]  # token per timestep

                    plt.figure(figsize=(12, 4))
                    plt.plot(forecast, label="Forecast")
                    forecast_len = len(forecast)
                    token_len = len(tokens)
                    positions = np.linspace(0, forecast_len - 1, token_len, dtype=int)
                    # print(forecast_len, token_len, len(positions))
                    segment_size = forecast.shape[0] // len(tokens)  # e.g. 672 // 16 = 42

                    for idx, word in enumerate(tokens):
                        center = idx * segment_size + segment_size // 2
                        if center < len(forecast) and word.strip():
                            plt.text(center, forecast[center], word, fontsize=7, rotation=45, ha='center')

                    plt.title(f"Forecast with Tokens (Sample {i * test_loader.batch_size + j})")
                    plt.xlabel("Timestep")
                    plt.ylabel("Value")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_path, f"forecast_tokens_{i}_{j}.png"))
                    plt.close()

        print(f"Visualizations saved to: {save_path}")
        return all_forecasts, all_tokens

