import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
class CrossStockAttention(nn.Module):
    """股票间交互注意力模块"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super(CrossStockAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, stock_features, stock_mask=None):
        # stock_features: [batch, num_stocks, d_model]
        # 股票间交互：每只股票都关注其他股票的特征
        key_padding_mask = None if stock_mask is None else ~stock_mask.bool()
        attended, _ = self.cross_attention(
            stock_features,
            stock_features,
            stock_features,
            key_padding_mask=key_padding_mask,
        )
        output = self.norm(stock_features + self.dropout(attended))
        return output

class FeatureAttention(nn.Module):
    """特征注意力模块"""
    def __init__(self, d_model, dropout=0.1):
        super(FeatureAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch*num_stocks, seq_len, d_model]
        attention_weights = self.attention(x)  # [batch*num_stocks, seq_len, 1]
        attended = torch.sum(x * attention_weights, dim=1)  # [batch*num_stocks, d_model]
        return self.dropout(attended)

class StockTransformer(nn.Module):
    def __init__(self, input_dim, config, num_stocks, emb_dim=None):
        super(StockTransformer, self).__init__()
        self.model_type = 'RankingTransformer'
        self.config = config
        self.num_stocks = num_stocks
        emb_dim = emb_dim or config.get('stock_embedding_dim', 16)
        self.id_dropout = config.get('id_dropout', 0.0)
        self.id_gate_enabled = bool(config.get('id_gate_enabled', False))

        # 0: padding，1: 未登录股票，已知股票从 2 开始编号。
        self.stock_embedding = nn.Embedding(num_stocks + 2, emb_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(config.get('embedding_dropout', 0.0))
        self.stock_embedding_proj = nn.Linear(emb_dim, config['d_model'])
        if self.id_gate_enabled:
            gate_init = float(config.get('id_gate_init', 0.20))
            if not 0.0 < gate_init < 1.0:
                raise ValueError('id_gate_init 必须位于 (0, 1)')
            self.identity_gate_logit = nn.Parameter(torch.tensor(
                math.log(gate_init / (1.0 - gate_init)),
                dtype=torch.float32,
            ))
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, config['d_model'])
        self.pos_encoder = PositionalEncoding(config['d_model'], config['dropout'], config['sequence_length'])
        
        # 时序特征提取
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['nhead'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config['num_layers'])
        
        # 特征注意力
        self.feature_attention = FeatureAttention(config['d_model'], config['dropout'])
        
        # 股票间交互注意力
        self.cross_stock_attention = CrossStockAttention(config['d_model'], config['nhead'], config['dropout'])
        
        # 排序特异性层
        self.ranking_layers = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model']),
            nn.LayerNorm(config['d_model']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.LayerNorm(config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        # 最终排序分数输出
        self.score_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1)
        )
        self.risk_heads_enabled = bool(
            config.get('risk_heads_enabled', False)
        )
        self.regime_gate_enabled = bool(
            config.get('regime_gate_enabled', False)
        )
        self.risk_penalty_scale = float(
            config.get('risk_penalty_scale', 0.25)
        )
        self.risk_1d_blend = float(config.get('risk_1d_blend', 0.40))
        self.risk_3d_blend = float(config.get('risk_3d_blend', 0.60))
        self.risk_5d_head_enabled = bool(
            config.get('risk_5d_head_enabled', False)
        )
        self.risk_5d_blend = float(
            config.get('risk_5d_blend', 0.0)
            if self.risk_5d_head_enabled else 0.0
        )
        self.tail_5d_head_enabled = bool(
            config.get('tail_5d_head_enabled', False)
        )
        self.tail_5d_blend = float(
            config.get('tail_5d_blend', 0.0)
            if self.tail_5d_head_enabled else 0.0
        )
        if self.risk_heads_enabled:
            if min(
                self.risk_1d_blend,
                self.risk_3d_blend,
                self.risk_5d_blend,
                self.tail_5d_blend,
            ) < 0:
                raise ValueError('风险头混合权重不能为负')
            blend_sum = (
                self.risk_1d_blend
                + self.risk_3d_blend
                + self.risk_5d_blend
                + self.tail_5d_blend
            )
            if blend_sum <= 0:
                raise ValueError('风险头混合权重和必须大于0')
            self.risk_1d_blend /= blend_sum
            self.risk_3d_blend /= blend_sum
            self.risk_5d_blend /= blend_sum
            self.tail_5d_blend /= blend_sum
            risk_head_input = config['d_model'] // 2
            risk_head_hidden = config['d_model'] // 4
            self.risk_1d_head = nn.Sequential(
                nn.Linear(risk_head_input, risk_head_hidden),
                nn.ReLU(),
                nn.Dropout(config['dropout'] * 0.5),
                nn.Linear(risk_head_hidden, 1),
            )
            self.risk_3d_head = nn.Sequential(
                nn.Linear(risk_head_input, risk_head_hidden),
                nn.ReLU(),
                nn.Dropout(config['dropout'] * 0.5),
                nn.Linear(risk_head_hidden, 1),
            )
            if self.risk_5d_head_enabled:
                self.risk_5d_head = nn.Sequential(
                    nn.Linear(risk_head_input, risk_head_hidden),
                    nn.ReLU(),
                    nn.Dropout(config['dropout'] * 0.5),
                    nn.Linear(risk_head_hidden, 1),
                )
            if self.tail_5d_head_enabled:
                self.tail_5d_head = nn.Sequential(
                    nn.Linear(risk_head_input, risk_head_hidden),
                    nn.ReLU(),
                    nn.Dropout(config['dropout'] * 0.5),
                    nn.Linear(risk_head_hidden, 1),
                )
        if self.regime_gate_enabled:
            regime_feature_indices = [
                int(index)
                for index in config.get(
                    'regime_market_feature_indices',
                    config.get('market_state_feature_indices', []),
                )
            ]
            if not regime_feature_indices:
                raise ValueError('市场状态门控缺少市场特征索引')
            if (
                min(regime_feature_indices) < 0
                or max(regime_feature_indices) >= input_dim
            ):
                raise ValueError('regime_market_feature_indices 超出输入维度')
            regime_hidden_size = int(
                config.get('regime_market_hidden_size', 16)
            )
            self.register_buffer(
                'regime_market_feature_indices',
                torch.tensor(regime_feature_indices, dtype=torch.long),
                persistent=False,
            )
            self.regime_market_encoder = nn.GRU(
                input_size=len(regime_feature_indices),
                hidden_size=regime_hidden_size,
                num_layers=1,
                batch_first=True,
            )
            self.regime_gate_head = nn.Sequential(
                nn.Linear(regime_hidden_size, regime_hidden_size),
                nn.ReLU(),
                nn.Dropout(config['dropout'] * 0.5),
                nn.Linear(regime_hidden_size, 1),
            )

        # 收益回归辅助头直接预测未来 5 日原始收益率。
        self.return_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1),
        )

        # Top-k 内相对仓位分配头；输出 logits，推理时经 softmax 后乘总仓位。
        self.allocation_head = nn.Sequential(
            nn.Linear(config['d_model'] // 2, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1),
        )

        # 市场级总仓位头。现金不单独建模，恒等于 1 - exposure。
        self.min_exposure = float(config.get('min_exposure', 0.80))
        self.max_exposure = float(config.get('max_exposure', 0.999999))
        if not 0.0 <= self.min_exposure < self.max_exposure < 1.0:
            raise ValueError('仓位范围必须满足 0 <= min_exposure < max_exposure < 1')
        self.exposure_market_encoder_enabled = bool(
            config.get('exposure_market_encoder_enabled', False)
        )
        exposure_input_dim = config['d_model'] // 2
        if self.exposure_market_encoder_enabled:
            market_feature_indices = [
                int(index)
                for index in config.get('market_state_feature_indices', [])
            ]
            if not market_feature_indices:
                raise ValueError(
                    '启用 Exposure 市场编码器时必须配置 market_state_feature_indices'
                )
            if min(market_feature_indices) < 0 or max(market_feature_indices) >= input_dim:
                raise ValueError('market_state_feature_indices 超出输入特征范围')
            market_hidden_size = int(
                config.get('exposure_market_hidden_size', 16)
            )
            if market_hidden_size < 1:
                raise ValueError('exposure_market_hidden_size 必须大于 0')
            self.register_buffer(
                'market_state_feature_indices',
                torch.tensor(market_feature_indices, dtype=torch.long),
                persistent=False,
            )
            self.exposure_market_encoder = nn.GRU(
                input_size=len(market_feature_indices),
                hidden_size=market_hidden_size,
                num_layers=1,
                batch_first=True,
            )
            exposure_input_dim += market_hidden_size
        self.exposure_portfolio_summary_enabled = bool(
            config.get('exposure_portfolio_summary_enabled', False)
        )
        self.monotonic_exposure_enabled = bool(
            config.get('monotonic_exposure_enabled', False)
        )
        if self.exposure_portfolio_summary_enabled:
            if not self.risk_heads_enabled or not self.regime_gate_enabled:
                raise ValueError(
                    'Exposure组合摘要需要启用风险头和市场状态门控'
                )
            # 单调模式下，压力和风险不再直接进入自由 MLP，只保留分数离散度。
            exposure_input_dim += (
                1 if self.monotonic_exposure_enabled else 5
            )
        if self.monotonic_exposure_enabled:
            if not self.risk_heads_enabled or not self.regime_gate_enabled:
                raise ValueError('单调 Exposure 需要风险头和市场状态门控')

            def inverse_softplus(value, name):
                value = float(value)
                if value <= 0:
                    raise ValueError(f'{name} 必须大于 0')
                return math.log(math.expm1(value))

            self.exposure_regime_penalty_raw = nn.Parameter(torch.tensor(
                inverse_softplus(
                    config.get('exposure_regime_penalty_init', 0.25),
                    'exposure_regime_penalty_init',
                ),
                dtype=torch.float32,
            ))
            self.exposure_risk_penalty_raw = nn.Parameter(torch.tensor(
                inverse_softplus(
                    config.get('exposure_risk_penalty_init', 0.25),
                    'exposure_risk_penalty_init',
                ),
                dtype=torch.float32,
            ))
        self.exposure_head = nn.Sequential(
            nn.Linear(exposure_input_dim, config['d_model'] // 4),
            nn.ReLU(),
            nn.Dropout(config['dropout'] * 0.5),
            nn.Linear(config['d_model'] // 4, 1),
        )
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def identity_gate_value(self):
        """返回ID分支的有效缩放；旧配置保持原来的1.0。"""
        if not self.id_gate_enabled:
            return self.stock_embedding.weight.new_tensor(1.0)
        return torch.sigmoid(self.identity_gate_logit)
    
    @staticmethod
    def _market_sequence(src, stock_mask, feature_indices):
        market_values = src.index_select(-1, feature_indices)
        if stock_mask is None:
            return market_values.mean(dim=1)
        market_mask = stock_mask.to(market_values.dtype)[:, :, None, None]
        return (
            (market_values * market_mask).sum(dim=1)
            / market_mask.sum(dim=1).clamp(min=1.0)
        )

    @staticmethod
    def _masked_cross_sectional_zscore(values, stock_mask):
        if stock_mask is None:
            return (
                values - values.mean(dim=1, keepdim=True)
            ) / values.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-6)
        mask = stock_mask.to(values.dtype)
        count = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean = (values * mask).sum(dim=1, keepdim=True) / count
        variance = (
            (values - mean).square() * mask
        ).sum(dim=1, keepdim=True) / count
        return (values - mean) / variance.sqrt().clamp(min=1e-6)

    def encode_backbone(self, src, stock_indices, stock_mask=None):
        """编码冻结的排序主干，供后续辅助阶段缓存复用。"""
        # src: [batch, num_stocks, seq_len, feature_dim]
        batch_size, num_stocks, seq_len, feature_dim = src.size()
        
        # 重塑为 [batch*num_stocks, seq_len, feature_dim]
        src_reshaped = src.view(batch_size * num_stocks, seq_len, feature_dim)
        
        # 输入投影和位置编码
        src_proj = self.input_proj(src_reshaped)  # [batch*num_stocks, seq_len, d_model]
        src_proj = self.pos_encoder(src_proj)
        
        # 时序特征提取
        temporal_features = self.temporal_encoder(src_proj)  # [batch*num_stocks, seq_len, d_model]
        
        # 特征注意力聚合
        aggregated_features = self.feature_attention(temporal_features)  # [batch*num_stocks, d_model]
        
        # 重塑回股票维度用于股票间交互
        stock_features = aggregated_features.view(batch_size, num_stocks, -1)  # [batch, num_stocks, d_model]
        embedding_indices = stock_indices
        if self.training and self.id_dropout > 0:
            # 随机将一部分已知股票替换为 UNK，抑制对单只股票身份的记忆。
            id_drop_mask = (
                (stock_indices > 1)
                & (torch.rand(stock_indices.shape, device=stock_indices.device) < self.id_dropout)
            )
            embedding_indices = stock_indices.masked_fill(id_drop_mask, 1)
        stock_embeddings = self.embedding_dropout(self.stock_embedding(embedding_indices))
        identity_features = self.stock_embedding_proj(stock_embeddings)
        stock_features = (
            stock_features
            + self.identity_gate_value() * identity_features
        )
        
        # 股票间交互注意力
        interactive_features = self.cross_stock_attention(
            stock_features,
            stock_mask=stock_mask,
        )  # [batch, num_stocks, d_model]
        
        # 重塑回原形状
        interactive_features = interactive_features.view(batch_size * num_stocks, -1)
        
        # 排序特异性变换
        ranking_features = self.ranking_layers(interactive_features)  # [batch*num_stocks, d_model//2]
        
        ranking_features = ranking_features.view(batch_size, num_stocks, -1)
        regime_sequence = None
        if self.regime_gate_enabled:
            regime_sequence = self._market_sequence(
                src, stock_mask, self.regime_market_feature_indices,
            )
        market_sequence = None
        if self.exposure_market_encoder_enabled:
            market_sequence = self._market_sequence(
                src.float(), stock_mask, self.market_state_feature_indices,
            )
        return ranking_features, regime_sequence, market_sequence

    def _forward_heads(
        self,
        ranking_features,
        stock_mask=None,
        regime_sequence=None,
        market_sequence=None,
        return_aux=False,
    ):
        """从排序主干表示运行各预测头；支持冻结主干缓存。"""
        batch_size, num_stocks, _ = ranking_features.shape
        flat_features = ranking_features.reshape(batch_size * num_stocks, -1)

        # 生成排序分数
        scores = self.score_head(flat_features)  # [batch*num_stocks, 1]
        predicted_returns = self.return_head(flat_features)  # [batch*num_stocks, 1]
        allocation_logits = self.allocation_head(flat_features)  # [batch*num_stocks, 1]
        
        # 重塑为最终输出格式
        raw_score_output = scores.view(batch_size, num_stocks)
        return_output = predicted_returns.view(batch_size, num_stocks)
        allocation_output = allocation_logits.view(batch_size, num_stocks)
        risk_1d_logits = None
        risk_3d_logits = None
        risk_5d_logits = None
        tail_5d_logits = None
        combined_risk = None
        if self.risk_heads_enabled:
            risk_1d_logits = self.risk_1d_head(flat_features).view(
                batch_size,
                num_stocks,
            )
            risk_3d_logits = self.risk_3d_head(flat_features).view(
                batch_size,
                num_stocks,
            )
            if self.risk_5d_head_enabled:
                risk_5d_logits = self.risk_5d_head(
                    flat_features
                ).view(batch_size, num_stocks)
            if self.tail_5d_head_enabled:
                tail_5d_logits = self.tail_5d_head(
                    flat_features
                ).view(batch_size, num_stocks)
            combined_risk = (
                self.risk_1d_blend * torch.sigmoid(risk_1d_logits)
                + self.risk_3d_blend * torch.sigmoid(risk_3d_logits)
            )
            if risk_5d_logits is not None:
                combined_risk = (
                    combined_risk
                    + self.risk_5d_blend * torch.sigmoid(risk_5d_logits)
                )
            if tail_5d_logits is not None:
                combined_risk = (
                    combined_risk
                    + self.tail_5d_blend * torch.sigmoid(tail_5d_logits)
                )

        regime_gate = raw_score_output.new_zeros(batch_size)
        if self.regime_gate_enabled:
            if regime_sequence is None:
                raise ValueError('市场状态门控缺少缓存序列')
            _, regime_hidden = self.regime_market_encoder(regime_sequence)
            regime_gate = torch.sigmoid(
                self.regime_gate_head(regime_hidden[-1])
            ).squeeze(-1)

        output = raw_score_output
        if (
            self.risk_heads_enabled
            and self.regime_gate_enabled
            and not self.config.get('oof_risk_penalty_enabled', False)
        ):
            output = (
                self._masked_cross_sectional_zscore(
                    raw_score_output,
                    stock_mask,
                )
                - self.risk_penalty_scale
                * regime_gate[:, None]
                * self._masked_cross_sectional_zscore(
                    combined_risk,
                    stock_mask,
                )
            )

        # Exposure 的 GRU、组合摘要和 BCE 输入保持 FP32。该分支很小，
        # 关闭 autocast 几乎不影响吞吐，但可避免 FP16 循环计算溢出。
        with torch.autocast(
            device_type=ranking_features.device.type,
            enabled=False,
        ):
            ranking_features_by_stock = ranking_features.float()
            if stock_mask is None:
                pooled_market_features = ranking_features_by_stock.mean(dim=1)
            else:
                valid_mask = stock_mask.to(
                    ranking_features_by_stock.dtype
                ).unsqueeze(-1)
                pooled_market_features = (
                    (ranking_features_by_stock * valid_mask).sum(dim=1)
                    / valid_mask.sum(dim=1).clamp(min=1.0)
                )
            exposure_features = pooled_market_features
            if self.exposure_market_encoder_enabled:
                if market_sequence is None:
                    raise ValueError('Exposure 市场编码器缺少缓存序列')
                _, market_hidden = self.exposure_market_encoder(
                    market_sequence
                )
                exposure_features = torch.cat(
                    [pooled_market_features, market_hidden[-1]],
                    dim=-1,
                )
            selected_combined_risk_mean = exposure_features.new_zeros(
                batch_size
            )
            if (
                self.exposure_portfolio_summary_enabled
                or self.monotonic_exposure_enabled
            ):
                top_k = min(5, num_stocks)
                selection_scores = output.float()
                if stock_mask is not None:
                    selection_scores = selection_scores.masked_fill(
                        ~stock_mask.bool(),
                        -torch.inf,
                    )
                top_indices = torch.topk(
                    selection_scores.detach(),
                    top_k,
                    dim=1,
                ).indices
                selected_risk_1d = torch.sigmoid(
                    risk_1d_logits.float()
                ).gather(1, top_indices)
                selected_risk_3d = torch.sigmoid(
                    risk_3d_logits.float()
                ).gather(1, top_indices)
                selected_combined_risk = combined_risk.float().gather(
                    1,
                    top_indices,
                )
                selected_combined_risk_mean = selected_combined_risk.mean(
                    dim=1
                )
                selected_scores = selection_scores.gather(1, top_indices)
                if self.exposure_portfolio_summary_enabled:
                    if self.monotonic_exposure_enabled:
                        portfolio_summary = selected_scores.std(
                            dim=1,
                            unbiased=False,
                        ).unsqueeze(1)
                    else:
                        portfolio_summary = torch.stack([
                            regime_gate.float(),
                            selected_risk_1d.mean(dim=1),
                            selected_risk_3d.mean(dim=1),
                            selected_combined_risk.max(dim=1).values,
                            selected_scores.std(dim=1, unbiased=False),
                        ], dim=1)
                    exposure_features = torch.cat(
                        [exposure_features, portfolio_summary],
                        dim=-1,
                    )
            exposure_logit = self.exposure_head(
                exposure_features
            ).squeeze(-1)
            base_exposure_probability = torch.sigmoid(exposure_logit)
            if self.monotonic_exposure_enabled:
                exposure_logit = (
                    exposure_logit
                    - F.softplus(self.exposure_regime_penalty_raw)
                    * regime_gate.float()
                    - F.softplus(self.exposure_risk_penalty_raw)
                    * selected_combined_risk_mean
                )
            raw_exposure = torch.sigmoid(exposure_logit)
            exposure = self.min_exposure + (
                self.max_exposure - self.min_exposure
            ) * raw_exposure
            exposure = exposure.clamp(
                min=self.min_exposure,
                max=self.max_exposure,
            )

        if return_aux:
            return (
                output,
                return_output,
                allocation_output,
                exposure,
                {
                    'raw_scores': raw_score_output,
                    'risk_1d_logits': risk_1d_logits,
                    'risk_3d_logits': risk_3d_logits,
                    'risk_5d_logits': risk_5d_logits,
                    'tail_5d_logits': tail_5d_logits,
                    'combined_risk': combined_risk,
                    'regime_gate': regime_gate,
                    'exposure_base_probability': base_exposure_probability,
                    'exposure_regime_penalty': (
                        F.softplus(self.exposure_regime_penalty_raw)
                        if self.monotonic_exposure_enabled else None
                    ),
                    'exposure_risk_penalty': (
                        F.softplus(self.exposure_risk_penalty_raw)
                        if self.monotonic_exposure_enabled else None
                    ),
                },
            )
        return output, return_output, allocation_output, exposure

    def forward_from_cached(
        self,
        ranking_features,
        regime_sequence=None,
        market_sequence=None,
        stock_mask=None,
        return_aux=False,
    ):
        return self._forward_heads(
            ranking_features,
            stock_mask=stock_mask,
            regime_sequence=regime_sequence,
            market_sequence=market_sequence,
            return_aux=return_aux,
        )

    def forward(self, src, stock_indices, stock_mask=None, return_aux=False):
        ranking_features, regime_sequence, market_sequence = self.encode_backbone(
            src, stock_indices, stock_mask,
        )
        return self._forward_heads(
            ranking_features,
            stock_mask=stock_mask,
            regime_sequence=regime_sequence,
            market_sequence=market_sequence,
            return_aux=return_aux,
        )
