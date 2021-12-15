import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.models as models


class TransZeroPP(nn.Module):
    def __init__(self, config):
        super(TransZeroPP, self).__init__()
        self.config = config
        self.dim_f = config.dim_f
        self.dim_v = config.dim_v
        self.nclass = config.num_class

        # bakcbone
        resnet101 = models.resnet101(pretrained=True)
        self.resnet101 = nn.Sequential(*list(resnet101.children())[:-2])

        # AVT
        self.transformer_s2v = TransformerPP(
            ec_layer=config.tf_ec_layer,
            dc_layer=config.tf_dc_layer,
            dim_com=config.tf_common_dim,
            dim_feedforward=config.tf_dim_feedforward,
            dropout=config.tf_dropout,
            SAtt=config.tf_SAtt,
            heads=config.tf_heads)
        # VAT
        self.transformer_v2s = TransformerPP(
            ec_layer=config.tf_ec_layer,
            dc_layer=config.tf_dc_layer,
            dim_com=config.tf_common_dim,
            dim_feedforward=config.tf_dim_feedforward,
            dropout=config.tf_dropout,
            SAtt=config.tf_SAtt,
            heads=config.tf_heads)
        
        self.V = nn.Parameter(torch.empty(
            config.num_attribute, self.dim_v), requires_grad=True)
        self.att = nn.Parameter(torch.empty(
            self.nclass, config.num_attribute), requires_grad=False)
        self.bias = nn.Parameter(torch.tensor(1), requires_grad=False)
        self.mask_bias = nn.Parameter(torch.empty(
            1, self.nclass), requires_grad=False)
        self.W_1_s2v = nn.Parameter(torch.empty(
            self.dim_v, config.tf_common_dim), requires_grad=True)
        self.W_3_s2v = nn.Parameter(torch.empty(
            self.dim_v, config.tf_common_dim), requires_grad=True)
        self.W_1_v2s = nn.Parameter(torch.empty(
            config.tf_common_dim, config.tf_common_dim), requires_grad=True)
        self.W_3_v2s = nn.Parameter(torch.empty(
            self.dim_f, config.tf_common_dim), requires_grad=True)
        self.W_4_v2s = nn.Parameter(torch.empty(
            config.tf_common_dim, config.tf_common_dim), requires_grad=True)

    def forward(self, imgs):
        Fs = self.resnet101(imgs)
        embed_s2v, embed_v2s = self.forward_feature_transformer(Fs)
        package = {'embed_s2v': self.forward_attribute(embed_s2v),
                   'embed_v2s': self.forward_attribute(embed_v2s)}
        package['embed'] = self.config.weight_s2v * package['embed_s2v'] + \
            (1 - self.config.weight_s2v) * package['embed_v2s']
        return package

    def forward_attribute(self, embed):
        embed = torch.einsum('ki,bi->bk', self.att, embed)
        self.vec_bias = self.mask_bias*self.bias
        embed = embed + self.vec_bias
        return embed

    def forward_feature_transformer(self, Fs):
        if len(Fs.shape) == 4:
            shape = Fs.shape
            Fs = Fs.reshape(shape[0], shape[1], shape[2] * shape[3])
        Fs = F.normalize(Fs, dim=1)
        Fs_pmt = Fs.permute(0, 2, 1)
        V_n = F.normalize(self.V) if self.config.normalize_V else self.V
        V_n_batch = V_n.unsqueeze(0).repeat(shape[0], 1, 1)
        # semantic-2-visual
        memory_s2v, _, emb_att_s2v = self.transformer_s2v.forward_encoder(
                Fs_pmt, V_n_batch)
        F_p_s2v = self.transformer_s2v.forward_decoder(
            memory_s2v, emb_att_s2v, type='s2v')
        S_p_s2v = torch.einsum('biv,vc,bic->bi', V_n_batch, self.W_1_s2v, F_p_s2v)
        embed_s2v = S_p_s2v
        # visual-2-semantic
        memory_v2s, emb_vis_v2s, emb_att_v2s = self.transformer_v2s.forward_encoder(
            Fs_pmt, V_n_batch)
        F_p_v2s = self.transformer_v2s.forward_decoder(
            memory_v2s, emb_att_v2s, type='v2s')
        S_p_v2s = torch.einsum('rbf,fc,brc->br', memory_v2s, self.W_1_v2s, F_p_v2s)
        E_v2s = torch.einsum('brc,cc,bic->bir', emb_vis_v2s, self.W_4_v2s, emb_att_v2s)
        embed_v2s = torch.einsum('bir,br->bi', E_v2s, S_p_v2s)

        return embed_s2v, embed_v2s
        

class Transformer(nn.Module):
    def __init__(self, ec_layer=1, dc_layer=1, dim_com=300,
                 dim_feedforward=2048, dropout=0.1, heads=1,
                 in_dim_cv=2048, in_dim_attr=300, SAtt=True):
        super(Transformer, self).__init__()
        # # input embedding
        self.embed_cv = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        self.embed_cv_aux = nn.Sequential(nn.Linear(in_dim_cv, dim_com))
        self.embed_attr = nn.Sequential(nn.Linear(in_dim_attr, dim_com))
        self.transformer_encoder = MultiLevelEncoder_woPad(N=ec_layer,
                                                           d_model=dim_com,
                                                           h=1,
                                                           d_k=dim_com,
                                                           d_v=dim_com,
                                                           d_ff=dim_feedforward,
                                                           dropout=dropout)
        decoder_layer = TransformerDecoderLayer(d_model=dim_com,
                                                nhead=heads,
                                                dim_feedforward=dim_feedforward,
                                                dropout=dropout,
                                                SAtt=SAtt)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=dc_layer)

    def forward(self, f_cv, f_attr):
        h_cv = self.embed_cv(f_cv.permute(0, 2, 1))
        h_attr = self.embed_attr(f_attr)
        h_attr_batch = h_attr.unsqueeze(0).repeat(f_cv.shape[0], 1, 1)
        memory = self.transformer_encoder(h_cv).permute(1, 0, 2)
        out = self.transformer_decoder(h_attr_batch.permute(1, 0, 2), memory)
        return out.permute(1, 0, 2)


class TransformerPP(Transformer):
    def __init__(self, ec_layer=1, dc_layer=1, dim_com=300,
                 dim_feedforward=2048, dropout=0.1, heads=1,
                 in_dim_cv=2048, in_dim_attr=300, SAtt=True):

        super(TransformerPP, self).__init__(
            ec_layer, dc_layer, dim_com, dim_feedforward, dropout,
            heads, in_dim_cv, in_dim_attr, SAtt)

    def forward_encoder(self, f_cv, f_attr, pre_embed=True, is_enc=True):
        if pre_embed:
            h_cv = self.embed_cv(f_cv)
            h_attr = self.embed_attr(f_attr)
        else:
            h_cv = f_cv
            h_attr = f_attr
        if is_enc:
            memory = self.transformer_encoder(h_cv).permute(1, 0, 2)
        return memory, h_cv, h_attr

    def forward_decoder(self, memory, h_attr, type='s2v'):
        if type == 's2v':
            out = self.transformer_decoder(h_attr.permute(1, 0, 2), memory)
        elif type == 'v2s':
            out = self.transformer_decoder(memory, h_attr.permute(1, 0, 2))
        return out.permute(1, 0, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(
            d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None, pos=None):
        q, k = (queries + pos, keys +
                pos) if pos is not None else (queries, keys)
        att = self.mhatt(q, k, values, relative_geometry_weights,
                         attention_mask, attention_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder_woPad(nn.Module):
    def __init__(self, N, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder_woPad, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])

        self.WGs = nn.ModuleList(
            [nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, input, attention_mask=None, attention_weights=None, pos=None):
        relative_geometry_embeddings = BoxRelationalEmbedding(
            input, grid_size=(14, 14))
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(
            -1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [layer(
            flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
        relative_geometry_weights = torch.cat(
            (relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)
        out = input
        for layer in self.layers:
            out = layer(out, out, out, relative_geometry_weights,
                        attention_mask, attention_weights, pos=pos)
        return out


class TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", SAtt=True):
        super(TransformerDecoderLayer, self).__init__(d_model, nhead,
                                                      dim_feedforward=dim_feedforward,
                                                      dropout=dropout,
                                                      activation=activation)
        self.SAtt = SAtt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.SAtt:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def get_relative_pos(x, batch_size, norm_len):
    x = x.view(1, -1, 1).expand(batch_size, -1, -1)
    return x / norm_len


def get_grids_pos(batch_size, seq_len, grid_size=(7, 7)):
    assert seq_len == grid_size[0] * grid_size[1]
    x = torch.arange(0, grid_size[0]).float().cuda()
    y = torch.arange(0, grid_size[1]).float().cuda()
    px_min = x.view(-1, 1).expand(-1, grid_size[0]).contiguous().view(-1)
    py_min = y.view(1, -1).expand(grid_size[1], -1).contiguous().view(-1)
    px_max = px_min + 1
    py_max = py_min + 1
    rpx_min = get_relative_pos(px_min, batch_size, grid_size[0])
    rpy_min = get_relative_pos(py_min, batch_size, grid_size[1])
    rpx_max = get_relative_pos(px_max, batch_size, grid_size[0])
    rpy_max = get_relative_pos(py_max, batch_size, grid_size[1])
    return rpx_min, rpy_min, rpx_max, rpy_max


def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True, grid_size=(7, 7)):
    batch_size, seq_len = f_g.size(0), f_g.size(1)
    x_min, y_min, x_max, y_max = get_grids_pos(batch_size, seq_len, grid_size)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)
    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)
    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))
    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)
    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))
        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(
            batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat
        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)


class ScaledDotProductGeometryAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, comment=None):
        super(ScaledDotProductGeometryAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()
        self.comment = comment

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, box_relation_embed_matrix, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h,
                                    self.d_k).permute(0, 2, 1, 3)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)
        v = self.fc_v(values).view(b_s, nk, self.h,
                                   self.d_v).permute(0, 2, 1, 3)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        w_g = box_relation_embed_matrix
        w_a = att
        w_mn = - w_g + w_a
        w_mn = torch.softmax(w_mn, -1)
        att = self.dropout(w_mn)
        out = torch.matmul(att, v).permute(
            0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)
        out = self.fc_o(out)
        return out


class MultiHeadGeometryAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None, comment=None):
        super(MultiHeadGeometryAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductGeometryAttention(
            d_model=d_model, d_k=d_k, d_v=d_v, h=h, comment=comment)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys
            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(
                q_norm, k_norm, v_norm, relative_geometry_weights, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(
                queries, keys, values, relative_geometry_weights, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=.1, identity_map_reordering=False):
        super(PositionWiseFeedForward, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, input):
        if self.identity_map_reordering:
            out = self.layer_norm(input)
            out = self.fc2(self.dropout_2(F.relu(self.fc1(out))))
            out = input + self.dropout(torch.relu(out))
        else:
            out = self.fc2(self.dropout_2(F.relu(self.fc1(input))))
            out = self.dropout(out)
            out = self.layer_norm(input + out)
        return out


if __name__ == '__main__':
    pass
