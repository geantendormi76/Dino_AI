# src/brain/decision_transformer.py
import torch
import torch.nn as nn
from src.brain.cnn_encoder import CNNEncoder

class DecisionTransformer(nn.Module):
    """
    决策Transformer模型。
    它将一系列的状态、动作和期望回报作为输入，来预测下一个最佳动作。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']

        # 1. 输入编码器 (Embeddings)
        # 使用CNN处理网格化状态
        self.state_encoder = CNNEncoder(
            in_channels=config['state_channels'], 
            output_dim=self.hidden_size
        )
        # 使用线性层处理回报和动作
        self.ret_to_go_encoder = nn.Linear(1, self.hidden_size)
        self.action_encoder = nn.Linear(config['action_dim'], self.hidden_size)
        
        # 位置编码
        self.position_embedding = nn.Embedding(config['max_len'] * 3, self.hidden_size) # 3 for (R, s, a)

        # 2. Transformer核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=config['n_head'],
            dim_feedforward=4 * self.hidden_size,
            dropout=config['dropout'],
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config['n_layer'])

        # 3. 输出头 (Prediction Heads)
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_size, config['action_dim']),
            nn.Tanh() # 假设动作被归一化到[-1, 1]
        )
        self.predict_action = nn.Linear(self.hidden_size, config['num_actions'])

    def forward(self, states, actions, returns_to_go, timesteps):
        batch_size, seq_len = states['arena_grid'].shape[0], states['arena_grid'].shape[1]

        # 编码输入
        state_embeddings = self.state_encoder(
            states['arena_grid'].reshape(-1, *states['arena_grid'].shape[2:])
        ).reshape(batch_size, seq_len, self.hidden_size)
        
        action_embeddings = self.action_encoder(actions)
        ret_embeddings = self.ret_to_go_encoder(returns_to_go)

        # 时间/位置编码
        # (B, T) -> (B, T, H)
        time_embeddings = self.position_embedding(timesteps.squeeze(-1))
        
        # 将位置编码加到每个token上
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        ret_embeddings = ret_embeddings + time_embeddings

        # 序列化: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # (B, T, H) -> (B, 3*T, H)
        stacked_inputs = torch.stack(
            (ret_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3 * seq_len, self.hidden_size)

        # 创建因果注意力掩码 (causal attention mask)
        # 防止模型在预测 t 时刻的动作时看到 t+1 时刻的信息
        mask = torch.nn.Transformer.generate_square_subsequent_mask(stacked_inputs.size(1))
        mask = mask.to(self.config['device'])

        # Transformer前向传播
        transformer_outputs = self.transformer(stacked_inputs, mask=mask)

        # 提取状态对应的输出token用于预测动作
        # 我们只关心输入's_t'后，模型预测的'a_t'
        state_outputs = transformer_outputs[:, 1::3, :] # (B, T, H)
        
        # 预测动作
        action_preds = self.predict_action(state_outputs) # (B, T, action_dim)

        return action_preds