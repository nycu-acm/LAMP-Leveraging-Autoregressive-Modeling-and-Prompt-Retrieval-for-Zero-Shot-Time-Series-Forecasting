import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
from transformers import GPT2Tokenizer
from layers.mlp import MLP
import os
import json

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

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        
        self.gpt2 = GPT2Model.from_pretrained(configs.llm_ckp_dir) 
        self.gpt2.to(self.device)
        self.hidden_dim_of_gpt2 = 768
        self.mix = configs.mix_embeds
        self.vocab_size = 50257

        self.word_embedding = self.gpt2.wte.state_dict()['weight'].to(device=self.device)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        
        for name, param in self.gpt2.named_parameters():
            param.requires_grad = False

        for name, param in self.gpt2.wte.named_parameters():
            param.requires_grad = False

        print_trainable_parameters(self.gpt2)

        if configs.mlp_hidden_layers == 0:
            print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2, 
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                               configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len,
                               configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                               configs.dropout, configs.mlp_activation)
        
        # Prompt Pool Initialization
        self.prompt_key_dict_trend = nn.ParameterDict({})
        self.prompt_value_dict_trend = nn.ParameterDict({})

        self.prompt_key_dict_seasonal = nn.ParameterDict({})
        self.prompt_value_dict_seasonal = nn.ParameterDict({})

        self.prompt_key_dict_residual = nn.ParameterDict({})
        self.prompt_value_dict_residual = nn.ParameterDict({})
        self.pool_size = 30
        self.top_k = 3
        self.prompt_len = 3


        self.mapping_value_layer = nn.Linear(self.vocab_size, self.prompt_len).to(self.device)
        self.mapping_key_layer = nn.Linear(self.vocab_size, 1).to(self.device)
        
        for i in range(self.pool_size):
            prompt_shape = (self.prompt_len, self.hidden_dim_of_gpt2)
            key_shape = (self.hidden_dim_of_gpt2,)
            
            self.prompt_value_dict_trend[f"prompt_value_{i}"] =  self.mapping_value_layer(self.word_embedding.permute(1, 0)).permute(1, 0).view(prompt_shape)
            self.prompt_key_dict_trend[f"prompt_key_{i}"] = self.mapping_key_layer(self.word_embedding.permute(1, 0)).permute(1, 0).squeeze().view(key_shape)
       
            self.prompt_value_dict_seasonal[f"prompt_value_{i}"] =  self.mapping_value_layer(self.word_embedding.permute(1, 0)).permute(1, 0).view(prompt_shape)
            self.prompt_key_dict_seasonal[f"prompt_key_{i}"] = self.mapping_key_layer(self.word_embedding.permute(1, 0)).permute(1, 0).squeeze().view(key_shape)

            self.prompt_value_dict_residual[f"prompt_value_{i}"] =  self.mapping_value_layer(self.word_embedding.permute(1, 0)).permute(1, 0).view(prompt_shape)
            self.prompt_key_dict_residual[f"prompt_key_{i}"] = self.mapping_key_layer(self.word_embedding.permute(1, 0)).permute(1, 0).squeeze().view(key_shape)

        if configs.data == 'ETTm1' or configs.data == 'ETTm2':
            prompt_file_path = os.path.join("prompt_bank", "prompt_bank", "ETTm_prompts.json")
        if configs.data == 'ETTh1' or configs.data == 'ETTh2':
            prompt_file_path = os.path.join("prompt_bank", "prompt_bank", "ETTh_prompts.json")
        elif configs.data == 'electricity':
            prompt_file_path = os.path.join("prompt_bank", "prompt_bank", "ECL_prompts.json")
        elif configs.data == 'weather':
            prompt_file_path = os.path.join("prompt_bank", "prompt_bank", "Weather_prompts.json")
        elif configs.data == 'traffic':
            prompt_file_path = os.path.join("prompt_bank", "prompt_bank", "Traffic_prompts.json")
        # Đọc prompt từ file
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            prompts = json.load(f)

        print("Using prompts from: ", prompt_file_path)

        self.gpt2_trend_token = self.tokenizer(text=prompts["trend"], return_tensors="pt").to(self.device)
        self.gpt2_season_token = self.tokenizer(text=prompts["seasonal"], return_tensors="pt").to(self.device)
        self.gpt2_residual_token = self.tokenizer(text=prompts["residual"], return_tensors="pt").to(self.device)
        self.hard_token_len = len(self.gpt2_trend_token['input_ids'][0])
        self.hard_trend_token_len = len(self.gpt2_trend_token['input_ids'][0])
        self.hard_seasonal_token_len = len(self.gpt2_season_token['input_ids'][0])
        self.hard_residual_token_len = len(self.gpt2_residual_token['input_ids'][0])

        with torch.no_grad():
            self.hard_trend_prompt = self.gpt2.wte(self.gpt2_trend_token['input_ids'])
        with torch.no_grad():
            self.hard_seasonal_prompt = self.gpt2.wte(self.gpt2_season_token['input_ids'])
        with torch.no_grad():
            self.hard_residual_prompt = self.gpt2.wte(self.gpt2_residual_token['input_ids'])


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def select_prompt(self, summary):
        prompt_key_matrix = torch.stack([self.prompt_key_dict[key] for key in self.prompt_key_dict])
        prompt_norm = self.l2_normalize(prompt_key_matrix, dim=1) # [pool_size, hidden_dim]
        summary_norm = self.l2_normalize(summary, dim=1)  # [batch_size, hidden_dim]
        
        similarity = torch.matmul(summary_norm, prompt_norm.t()) # [batch_size, pool_size]
        topk_sim, idx = torch.topk(similarity, k=self.top_k, dim=1)
        
        prompt_value_matrix = torch.stack([self.prompt_value_dict[key] for key in self.prompt_value_dict])
        batched_prompt_raw = prompt_value_matrix[idx]  # [batch_size, top_k, prompt_len, hidden_dim]
        batch_size, top_k, length, c = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.view(batch_size, top_k * length, c) # [batch_size, top_k * prompt_len, hidden_dim]
        
        return batched_prompt


    
    def select_prompt_by_type(self, summary, prompt_key_dict, prompt_value_dict):
        prompt_key_matrix = torch.stack([prompt_key_dict[key] for key in prompt_key_dict])
        prompt_norm = self.l2_normalize(prompt_key_matrix, dim=1)
        summary_norm = self.l2_normalize(summary, dim=1)
        
        similarity = torch.matmul(summary_norm, prompt_norm.t())
        _, idx = torch.topk(similarity, k=self.top_k, dim=1)

        prompt_value_matrix = torch.stack([prompt_value_dict[key] for key in prompt_value_dict])
        batched_prompt_raw = prompt_value_matrix[idx]
        batch_size, top_k, length, c = batched_prompt_raw.shape

        batched_key_norm = prompt_norm[idx]
        summary_embed_norm = summary.unsqueeze(1)
        sim = batched_key_norm * summary_embed_norm
        reduce_sim = torch.sum(sim) / summary.shape[0]


        return batched_prompt_raw.view(batch_size, top_k * length, c), reduce_sim
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, trend=None, seasonal=None, residual=None):
        
        bs, _, n_vars = x_enc.shape

        means_trend = trend.mean(1, keepdim=True).detach()
        trend = trend - means_trend
        stdev_trend = torch.sqrt(
            torch.var(trend, dim=1, keepdim=True, unbiased=False) + 1e-5)
        trend /= stdev_trend
        trend = trend.permute(0, 2, 1).reshape(bs * n_vars, -1)
        fold_out_trend = trend.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num_trend = fold_out_trend.shape[1]

        means_seasonal = seasonal.mean(1, keepdim=True).detach()
        seasonal = seasonal - means_seasonal
        stdev_seasonal = torch.sqrt(
            torch.var(seasonal, dim=1, keepdim=True, unbiased=False) + 1e-5)
        seasonal /= stdev_seasonal
        seasonal = seasonal.permute(0, 2, 1).reshape(bs * n_vars, -1)
        fold_out_seasonal = seasonal.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num_seasonal = fold_out_seasonal.shape[1]

        means_residual = residual.mean(1, keepdim=True).detach()
        residual = residual - means_residual
        stdev_residual = torch.sqrt(
            torch.var(residual, dim=1, keepdim=True, unbiased=False) + 1e-5)
        residual /= stdev_residual
        residual = residual.permute(0, 2, 1).reshape(bs * n_vars, -1)
        fold_out_residual = residual.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num_residual = fold_out_residual.shape[1]

        trend_embeds = self.encoder(fold_out_trend)  # [bs * n_vars, token_num_trend, hidden_dim]
        seasonal_embeds = self.encoder(fold_out_seasonal)  # [bs * n_vars, token_num_seasonal, hidden_dim]
        residual_embeds = self.encoder(fold_out_residual)  # [bs * n_vars, token_num_residual, hidden_dim]


        summary_trend = trend_embeds.mean(dim=1)
        prompt_embeds_trend, simlarity_loss_trend = self.select_prompt_by_type(summary_trend, self.prompt_key_dict_trend, self.prompt_value_dict_trend)
        hard_trend_prompt = self.hard_trend_prompt.repeat(bs * n_vars, 1, 1)
        trend_embeds = torch.cat((hard_trend_prompt, prompt_embeds_trend, trend_embeds), dim=1)  # [bs * n_vars, new_seq_len, hidden_dim]
        summary_seasonal = seasonal_embeds.mean(dim=1)
        prompt_embeds_seasonal, simlarity_loss_seasonal = self.select_prompt_by_type(summary_seasonal, self.prompt_key_dict_seasonal, self.prompt_value_dict_seasonal)
        hard_seasonal_prompt = self.hard_seasonal_prompt.repeat(bs * n_vars, 1, 1)
        seasonal_embeds = torch.cat((hard_seasonal_prompt, prompt_embeds_seasonal, seasonal_embeds), dim=1)  # [bs * n_vars, new_seq_len, hidden_dim]
        summary_residual = residual_embeds.mean(dim=1)
        prompt_embeds_residual, similarity_loss_residual = self.select_prompt_by_type(summary_residual, self.prompt_key_dict_residual, self.prompt_value_dict_residual)
        hard_residual_prompt = self.hard_residual_prompt.repeat(bs * n_vars, 1, 1)
        residual_embeds = torch.cat((hard_residual_prompt, prompt_embeds_residual, residual_embeds), dim=1)  # [bs * n_vars, new_seq_len, hidden_dim]
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        
        # Select prompt embeddings and concatenate
        times_embeds_concat = torch.cat((trend_embeds, seasonal_embeds, residual_embeds), dim=1)  # [bs * n_vars, new_seq_len, hidden_dim]
        outputs_concat = self.gpt2(inputs_embeds=times_embeds_concat).last_hidden_state
        outputs_trend = outputs_concat[:, :token_num_trend+prompt_embeds_trend.shape[1]+self.hard_trend_token_len,:]
        outputs_seasonal = outputs_concat[:, token_num_trend+prompt_embeds_trend.shape[1]+self.hard_trend_token_len:token_num_trend*2+2*prompt_embeds_trend.shape[1]+self.hard_trend_token_len+self.hard_seasonal_token_len,:]
        outputs_residual = outputs_concat[:, 2*token_num_trend+2*prompt_embeds_trend.shape[1]+self.hard_trend_token_len+self.hard_seasonal_token_len:, :]

        dec_out_trend = self.decoder(outputs_trend[:, prompt_embeds_trend.shape[1]+self.hard_trend_token_len:])
        dec_out_trend = dec_out_trend.reshape(bs, n_vars, -1)
        dec_out_trend = dec_out_trend.permute(0, 2, 1)
        dec_out_trend = dec_out_trend * \
                (stdev_trend[:, 0, :].unsqueeze(1).repeat(1, token_num_trend * self.token_len, 1))
        dec_out_trend = dec_out_trend + \
            (means_trend[:, 0, :].unsqueeze(1).repeat(1, token_num_trend * self.token_len, 1))
        
        dec_out_seasonal = self.decoder(outputs_seasonal[:, prompt_embeds_seasonal.shape[1]+self.hard_seasonal_token_len:])
        dec_out_seasonal = dec_out_seasonal.reshape(bs, n_vars, -1)
        dec_out_seasonal = dec_out_seasonal.permute(0, 2, 1)
        dec_out_seasonal = dec_out_seasonal * \
                (stdev_seasonal[:, 0, :].unsqueeze(1).repeat(1, token_num_seasonal * self.token_len, 1))
        dec_out_seasonal = dec_out_seasonal + \
            (means_seasonal[:, 0, :].unsqueeze(1).repeat(1, token_num_seasonal * self.token_len, 1))
        
        dec_out_residual = self.decoder(outputs_residual[:, prompt_embeds_residual.shape[1]+self.hard_residual_token_len:])
        dec_out_residual = dec_out_residual.reshape(bs, n_vars, -1)
        dec_out_residual = dec_out_residual.permute(0, 2, 1)
        dec_out_residual = dec_out_residual * \
                (stdev_residual[:, 0, :].unsqueeze(1).repeat(1, token_num_residual * self.token_len, 1))
        dec_out_residual = dec_out_residual + \
            (means_residual[:, 0, :].unsqueeze(1).repeat(1, token_num_residual * self.token_len, 1))
        dec_out_new = dec_out_trend + dec_out_seasonal + dec_out_residual

        def orthogonal_constraint(x1, x2, x3):
            
            x1 = x1.view(x1.size(0), -1)
            x2 = x2.view(x2.size(0), -1)
            x3 = x3.view(x3.size(0), -1)

            inner_product_12 = torch.sum(x1 * x2, dim=1)
            inner_product_13 = torch.sum(x1 * x3, dim=1)
            inner_product_23 = torch.sum(x2 * x3, dim=1)

            loss = torch.mean(inner_product_12**2 + inner_product_13**2 + inner_product_23**2)
            
            return loss
        
        orthogonal_loss = orthogonal_constraint(prompt_embeds_trend, prompt_embeds_seasonal, prompt_embeds_residual)
        simlarity_loss = simlarity_loss_trend + simlarity_loss_seasonal + similarity_loss_residual

        return dec_out_new, orthogonal_loss, simlarity_loss
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, trend=None, seasonal=None, residual=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, trend, seasonal, residual)
    
    def forecast_return_gpt2(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).reshape(bs * n_vars, -1)
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        times_embeds = self.encoder(fold_out)  # [bs * n_vars, token_num, hidden_dim]

        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc

        summary = times_embeds.mean(dim=1)
        prompt_embeds = self.select_prompt(summary)
        times_embeds = torch.cat((prompt_embeds, times_embeds), dim=1)

        gpt2_outputs = self.gpt2(inputs_embeds=times_embeds).last_hidden_state

        dec_out = self.decoder(gpt2_outputs[:, prompt_embeds.shape[1]:])
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))

        return dec_out, gpt2_outputs
