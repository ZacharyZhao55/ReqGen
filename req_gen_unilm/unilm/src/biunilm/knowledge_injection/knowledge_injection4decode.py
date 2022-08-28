import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import os

from random import randint, shuffle, choice
from random import random as rand

from loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

import math
import numpy as np
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, BertLayerNorm
from pytorch_pretrained_bert.tokenization import whitespace_tokenize, BasicTokenizer
from transformers import BertTokenizer
import re


def sort_batch(data, label, length):
    batch_size = data.size(0)
    # 先将数据转化为numpy()，再得到排序的index
    inx = torch.from_numpy(np.argsort(length.numpy())[::-1].copy())
    data = data[inx]
    label = label[inx]
    length = length[inx]
    # length转化为了list格式，不再使用torch.Tensor格式
    length = list(length.numpy())
    return (data, label, length)


def reverse_padded_sequence(inputs, lengths, batch_first=True):
    '''这个函数输入是Variable，在Pytorch0.4.0中取消了Variable，输入tensor即可
    '''
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (Variable): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A Variable with the same size as inputs, but with each sequence
        reversed according to its length.
    """
    if batch_first:
        inputs = inputs.transpose(0, 1)
    max_length, batch_size = inputs.size(0), inputs.size(1)
    if len(lengths) != batch_size:
        raise ValueError("inputs is incompatible with lengths.")
    ind = [list(reversed(range(0, length))) + list(range(length, max_length))
           for length in lengths]
    ind = torch.LongTensor(ind).transpose(0, 1)
    for dim in range(2, inputs.dim()):
        ind = ind.unsqueeze(dim)
    ind = Variable(ind.expand_as(inputs))
    if inputs.is_cuda:
        ind = ind.cuda(inputs.get_device())
    reversed_inputs = torch.gather(inputs, 0, ind)
    if batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, biFlag, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        if (biFlag):
            self.bi_num = 2
        else:
            self.bi_num = 1
        self.biFlag = biFlag

        self.layer1 = nn.ModuleList()
        self.layer1.append(nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, \
                                   num_layers=num_layers, batch_first=True, \
                                   dropout=dropout, bidirectional=0))
        if (biFlag):
            # 如果是双向，额外加入逆向层
            self.layer1.append(nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, \
                                       num_layers=num_layers, batch_first=True, \
                                       dropout=dropout, bidirectional=0))

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim * self.bi_num, output_dim),
            nn.LogSoftmax(dim=2)
        )

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device),
                torch.zeros(self.num_layers * self.bi_num, batch_size, self.hidden_dim).to(self.device))

    def forward(self, x, y, length):
        batch_size = x.size(0)
        max_length = torch.max(length)
        x = x[:, 0:max_length, :];
        y = y[:, 0:max_length]
        x, y, length = sort_batch(x, y, length)
        x, y = x.to(self.device), y.to(self.device)
        hidden = [self.init_hidden(batch_size) for l in range(self.bi_num)]

        out = [x, reverse_padded_sequence(x, length, batch_first=True)]
        for l in range(self.bi_num):
            # pack sequence
            out[l] = nn.utils.rnn.pack_padded_sequence(out[l], length, batch_first=True)
            out[l], hidden[l] = self.layer1[l](out[l], hidden[l])
            # unpack
            out[l], _ = nn.utils.rnn.pack_padded_sequence(out[l], batch_first=True)
            # 如果是逆向层，需要额外将输出翻过来
            if (l == 1): out[l] = reverse_padded_sequence(out[l], length, batch_first=True)

        if (self.bi_num == 1):
            out = out[0]
        else:
            out = torch.cat(out, 2)
        out = self.layer2(out)
        out = torch.squeeze(out)
        return y, out, length


class DotProductAttention(nn.Module):
    def __init__(self, keys_size, queries_size, num_hiddens, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.W_q = nn.Linear(queries_size, num_hiddens, bias=False).to(device)
        self.W_k = nn.Linear(keys_size, num_hiddens, bias=False).to(device)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, queries, keys, values, attention_mask):

        d = queries.shape[-1]
        '''交换keys的后两个维度，相当于公式中的转置'''
        # tem_attention_mask = torch.bmm(attention_mask.squeeze(), attention_mask.squeeze().transpose(1, 2))
        # attention_mask = attention_mask[-1].reshape(keys.shape).permute(2, 0, 1)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        # scores = scores + attention_mask.expand_as(scores)
        #[64, 16, 2]
        # attention_mask = attention_mask.reshape(-1,1,8,1024)
        # for i in attention_mask:
        #     scores = scores + torch.bmm(queries, i.expand_as(keys).transpose(1, 2))
        self.attention_weights = F.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)


class Encoder(nn.Module):
    def __init__(self, inputs_dim, num_hiddens, hiddens_layers):
        super(Encoder, self).__init__()
        # self.rnn1 = nn.LSTM(
        #     input_size=inputs_dim, hidden_size=num_hiddens, bidirectional=True)
        self.rnn1 = nn.LSTM(
            input_size=inputs_dim, hidden_size=num_hiddens,
            num_layers=hiddens_layers,bidirectional=True)
        self.knowledge_encoder = nn.LSTM(input_size=inputs_dim, hidden_size=num_hiddens,
                                           bidirectional=True)

    def forward(self, inputs):
        '''由于nn.GRU没有设置 batch_first=True
           因此输入的维度排列：[time_step_num, batch_size, num_features]
           输出维度为：
                output: [time_step_num, batch_size, hiddens_num]
                hidSta: [num_layers, batch_size, hiddens_num]
        '''
        inputs = inputs.permute(0, 1, 2)
        # encOut, hidSta = self.rnn1(inputs)
        lstm_output, (h, c) = self.rnn1(inputs)
        return lstm_output, (h, c)


class Attention(nn.Module):
    def __init__(
            self, inputs_dim, num_hiddens, num_layers, outputs_dim, dropout):
        super(Attention, self).__init__()
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.attention = DotProductAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout).to(device)
        self.rnn = nn.GRU(
            num_hiddens, num_hiddens, num_layers,
            dropout=dropout).to(device)
        self.wa = torch.nn.Linear(768, 768).to(device)

        layer_norm_eps = 1e-5
        self.LayerNorm = BertLayerNorm(num_hiddens*2, eps=layer_norm_eps).to(device)

    def forward(self, unilm_output, enc_outputs, attention_mask):
        '''
        inputs: [batch_size, time_step_num, features]
        states:
            enc_ouptut, enc_hidden_state
        '''
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # enc_outputs, hidden_state = states
        '''将enc_output的维度变为[batch_size, time_step_num, enc_hidden_num]'''
        enc_outputs = enc_outputs.to(device)
        output_last = enc_outputs.permute(1, 0, 2)

        unilm_output = unilm_output.to(device)
        unilm_output = unilm_output.permute(1, 0, 2)

        '''将inputs的维度变为[time_step_num, batch_size, features_num]'''
        outputs, self._attention_weights = [], []
        '''对每一时间步的inputs进行计算，并于上下文信息进行融合'''
        query = unilm_output
        context = self.attention(query, output_last, output_last, attention_mask)
        x_result = query + self.wa(context)

        x_result = x_result.permute(1, 0, 2)

        return self.LayerNorm(x_result)

class AttentionDAF(nn.Module):
    def __init__(
            self, inputs_dim, num_hiddens, num_layers, outputs_dim, dropout):
        super(AttentionDAF, self).__init__()
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.attention = DotProductAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout).to(device)
        self.rnn = nn.GRU(
            num_hiddens, num_hiddens, num_layers,
            dropout=dropout).to(device)
        self.relu = nn.ReLU()
        # self.rnn = nn.GRU(
        #     inputs_dim + num_hiddens, num_hiddens, num_layers,
        #     dropout=dropout).to(device)
        # self.rnn = nn.GRU(
        #     inputs_dim, num_hiddens, num_layers,
        #     dropout=dropout)
        self.dense = nn.Linear(num_hiddens, outputs_dim).to(device)
        self.att_weight_c = nn.Linear(num_hiddens * 2, 1).to(device)
        self.att_weight_q = nn.Linear(num_hiddens * 2, 1).to(device)
        self.att_weight_cq = nn.Linear(num_hiddens * 2, 1).to(device)
        self.modeling_LSTM2 = nn.LSTM(input_size=num_hiddens * 4,
                                   hidden_size=num_hiddens,
                                   bidirectional=False,
                                   batch_first=True,
                                   dropout=dropout).to(device)
        self.modeling_LSTM1 = nn.LSTM(input_size=num_hiddens,
                                   hidden_size=num_hiddens,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout).to(device)
        self.wa = torch.nn.Linear(3072, 768).to(device)
        # self.wa = torch.nn.Linear(2048, 1024).to(device)
        self.relu = nn.ReLU()
        layer_norm_eps = 1e-5
        self.LayerNorm = BertLayerNorm(num_hiddens*2, eps=layer_norm_eps).to(device)


    def forward(self, inputs, states, attention_mask):
        '''
            :param c: (batch, c_len, hidden_size * 2)
            :param q: (batch, q_len, hidden_size * 2)
            :return: (batch, c_len, q_len)
        '''
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        c_len = inputs.size(1)
        q_len = states.size(1)
        c = inputs.to(device)
        q = states.to(device)


        cq = []
        for i in range(q_len):
            # (batch, 1, hidden_size * 2)
            qi = q.select(1, i).unsqueeze(1)
            # (batch, c_len, 1)
            ci = self.att_weight_cq(c * qi).squeeze()
            cq.append(ci)
        # (batch, c_len, q_len)
        cq = torch.stack(cq, dim=-1)

        # (batch, c_len, q_len)
        s = self.att_weight_c(c).expand(-1, -1, q_len) + \
            self.att_weight_q(q).permute(0, 2, 1).expand(-1, c_len, -1) + \
            cq

        # '''add mask information'''
        # attention_mask = attention_mask.permute(1, 0, 2, 3)
        # for i in attention_mask:
        #     s += i
        # ''''''
        s = s + attention_mask.squeeze(1)

        # (batch, c_len, q_len)
        a = F.softmax(s, dim=2)
        # (batch, c_len, q_len) * (batch, q_len, hidden_size * 2) -> (batch, c_len, hidden_size * 2)
        c2q_att = torch.bmm(a, q)
        # (batch, 1, c_len)
        b = F.softmax(torch.max(s, dim=2)[0], dim=1).unsqueeze(1)
        # (batch, 1, c_len) * (batch, c_len, hidden_size * 2) -> (batch, hidden_size * 2)
        q2c_att = torch.bmm(b, c).squeeze()
        if len(q2c_att.shape) == 1:
            q2c_att = q2c_att.unsqueeze(0)
        # (batch, c_len, hidden_size * 2) (tiled)
        q2c_att = q2c_att.unsqueeze(1).expand(-1, c_len, -1)
        # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        # (batch, c_len, hidden_size * 8)
        # x = torch.cat([c, c2q_att], dim=-1).to(device)
        # x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1).to(device)
        x = torch.cat([c, c2q_att, c * c2q_att, c * q2c_att], dim=-1).to(device)

        # m = self.modeling_LSTM2(x)

        x = self.relu(self.wa(x))

        # return self.relu(torch.bmm(x,wa))
        return self.LayerNorm(x+c)
        # return x

def filter_n_hop_rel(sen):
    rel_list = sen.strip().split('.')
    filter_result = []
    for rel in rel_list:
        # print(rel)
        pattern = '#(.+?)#'
        hop = re.findall(pattern, rel)
        if len(hop) > 0:
            hop_value = int(hop[0])
        else:
            hop_value = 1
        if hop_value == 1:
            filter_result.append(rel)
        elif hop_value == 2 and rand() < 0.5:
            filter_result.append(rel)
        elif hop_value == 3 and rand() < 0.25:
            filter_result.append(rel)
        elif hop_value == 4 and rand() < 0.1:
            filter_result.append(rel)
        elif hop_value == 5 and rand() < 0.05:
            filter_result.append(rel)
    filter_sen = ' '.join(filter_result)
    return filter_sen

class knowledge_injection(nn.Module):
    def __init__(self, model_name, inputs_dim, num_hiddens, enc_hiddens_layers, att_hiddens_layers, dropout,
                 output_dim):
        super(knowledge_injection, self).__init__()
        self.model_name = model_name
        self.inputs_dim = inputs_dim
        self.num_hiddens = num_hiddens
        self.enc_hiddens_layers = enc_hiddens_layers
        self.att_hiddens_layers = att_hiddens_layers
        self.dropout = dropout
        self.output_dim = output_dim
        self.encoder = Encoder(inputs_dim=self.inputs_dim, num_hiddens=self.num_hiddens,
                          hiddens_layers=self.enc_hiddens_layers)

        self.model = BertModel.from_pretrained(self.model_name)

        self.attention = Attention(inputs_dim=self.inputs_dim, num_hiddens=self.num_hiddens,
                              num_layers=self.att_hiddens_layers,
                              outputs_dim=self.output_dim, dropout=self.dropout)


    def forward(self, sentences, unilm_outputs, unilm_mask):
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        bert_tokenizer = BertTokenizer.from_pretrained(self.model_name)
        extended_attention_mask = None
        input_ids = None
        max_length = unilm_outputs.shape[1]

        for sen in sentences:
            sen = filter_n_hop_rel(sen)
            pattern = '#(.+?)#'
            sen_no_hop_label = re.sub(pattern, '', sen)

            sen_token = bert_tokenizer.tokenize(text=sen_no_hop_label)

            max_len = max_length
            mode = 's2s'
            # mode = None
            self._tril_matrix = torch.tril(torch.ones(
                (max_len, max_len), dtype=torch.long))
            input_mask = torch.zeros(max_len, max_len, dtype=torch.long)
            if mode == "s2s":
                input_mask[:, :len(sen_token) + 2].fill_(1)

            else:
                st, end = 0, len(sen_token) + 3
                input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])

            if input_ids == None:
                input_ids = bert_tokenizer.encode(
                    sen_no_hop_label,
                    add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                    max_length=max_length,  # 设定最大文本长度
                    pad_to_max_length=True,  # pad到最大的长度
                    return_tensors='pt',  # 返回的类型为pytorch tensor
                    truncation = True
                )
                extended_attention_mask = self.model.get_extended_attention_mask(input_ids, token_type_ids=None, attention_mask=input_mask).permute(2,1,0,3)
            else:
                new_input_ids = bert_tokenizer.encode(
                    sen_no_hop_label,
                    add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                    max_length=max_length,  # 设定最大文本长度
                    pad_to_max_length=True,  # pad到最大的长度
                    return_tensors='pt',  # 返回的类型为pytorch tensor
                    truncation=True
                )
                input_ids = torch.cat((input_ids,new_input_ids),dim=0)
                new_extended_attention_mask = self.model.get_extended_attention_mask(new_input_ids, token_type_ids=None, attention_mask=input_mask).permute(2,1,0,3)
                extended_attention_mask = torch.cat((extended_attention_mask,new_extended_attention_mask),dim=0)

        unilm_mask = unilm_mask.to(device)
        # extended_attention_mask  = rel_attention_mask.to(device)
        # joint_attention_mask = torch.cat((unilm_mask.expand_as(extended_attention_mask),extended_attention_mask), dim=1)
        # joint_attention_mask = joint_attention_mask.permute(2,1,0)
        #[16, 1, 64, 64]
        joint_attention_mask = extended_attention_mask + unilm_mask
        input_ids = input_ids.to(device)
        inp_ebd = self.model.embeddings(input_ids, token_type_ids=None, task_idx=None)
        enc_output, (h,c) = self.encoder(inp_ebd)

        dec_output = self.attention(unilm_outputs, enc_output, joint_attention_mask)
        return dec_output