import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch import nn
import numpy as np
import math
from torch.nn import init
from transformers.models.ctrl.modeling_ctrl import MultiHeadAttention

from models.attention import SimpleEncoder
from torch.nn.utils import rnn
from transformers import *
from torch.nn.utils.rnn import pad_sequence


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config

        word_vec_size = config.data_word_vec.shape[0]
        self.word_emb = nn.Embedding(word_vec_size, config.data_word_vec.shape[1])
        self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec))

        self.word_emb.weight.requires_grad = False
        self.use_entity_type = True
        self.use_coreference = True
        self.use_distance = True

        # hidden_size = 128

        hidden_size = 768
        self.self_attention = SelfAttention(hidden_size)

        bert_hidden_size = 768
        input_size = config.data_word_vec.shape[1]
        if self.use_entity_type:
            input_size += config.entity_type_size
            self.ner_emb = nn.Embedding(7, config.entity_type_size, padding_idx=0)

        if self.use_coreference:
            input_size += config.coref_size
            self.entity_embed = nn.Embedding(config.max_length, config.coref_size, padding_idx=0)


        self.input_size = input_size
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear_re = nn.Linear(bert_hidden_size, hidden_size)

        if self.use_distance:
            self.dis_embed = nn.Embedding(20, config.dis_size, padding_idx=10)
            self.bili = torch.nn.Bilinear(hidden_size+config.dis_size, hidden_size+config.dis_size, config.relation_num)
        else:
            self.bili = torch.nn.Bilinear(hidden_size, hidden_size, config.relation_num)

    def mask_lengths(self, batch_size, doc_size, lengths):
        masks = torch.ones(batch_size, doc_size).cuda()
        index_matrix = torch.arange(0, doc_size).expand(batch_size, -1).cuda()
        index_matrix = index_matrix.long()
        doc_lengths = lengths.view(-1, 1)
        doc_lengths_matrix = doc_lengths.expand(-1, doc_size)
        masks[torch.ge(index_matrix-doc_lengths_matrix, 0)] = 0
        return masks

    def forward(self, context_idxs, pos, context_ner, context_char_idxs, context_lens, h_mapping, t_mapping,
                relation_mask, dis_h_2_t, dis_t_2_h, sent_idxs, sent_lengths, reverse_sent_idxs, context_masks, context_starts):
        context_output = self.bert(context_idxs, attention_mask=context_masks)[0]
                # 应用自注意力机制
        context_output = self.self_attention(context_output)
        context_output = [layer[starts.nonzero().squeeze(1)]
                   for layer, starts in zip(context_output, context_starts)]
        context_output = pad_sequence(context_output, batch_first=True, padding_value=-1)

        context_output = torch.nn.functional.pad(context_output,(0,0,0,context_idxs.size(-1)-context_output.size(-2)))

        context_output = self.linear_re(context_output)


        start_re_output = torch.matmul(h_mapping, context_output)
        end_re_output = torch.matmul(t_mapping, context_output)


        if self.use_distance:
            s_rep = torch.cat([start_re_output, self.dis_embed(dis_h_2_t)], dim=-1)
            t_rep = torch.cat([end_re_output, self.dis_embed(dis_t_2_h)], dim=-1)

            predict_re = self.bili(s_rep, t_rep)
        else:

            predict_re = self.bili(start_re_output, end_re_output)

        return predict_re


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        dropout = self.dropout
        if not self.training:
            return x
        m = x.data.new(x.size(0), 1, x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m.div_(1 - dropout), requires_grad=False)
        mask = mask.expand_as(x)
        return mask * x

class EncoderRNN(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.GRU(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)
        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        if input_lengths is not None:
            lens = input_lengths.data.cpu().numpy()
        for i in range(self.nlayers):
            hidden = self.get_init(bsz, i)
            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, hidden = self.rnns[i](output, hidden)


            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        if self.concat:
            return torch.cat(outputs, dim=2)
        return outputs[-1]




class EncoderLSTM(nn.Module):
    def __init__(self, input_size, num_units, nlayers, concat, bidir, dropout, return_last):
        super().__init__()
        self.rnns = []
        for i in range(nlayers):
            if i == 0:
                input_size_ = input_size
                output_size_ = num_units
            else:
                input_size_ = num_units if not bidir else num_units * 2
                output_size_ = num_units
            self.rnns.append(nn.LSTM(input_size_, output_size_, 1, bidirectional=bidir, batch_first=True))
        self.rnns = nn.ModuleList(self.rnns)

        self.init_hidden = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])
        self.init_c = nn.ParameterList([nn.Parameter(torch.Tensor(2 if bidir else 1, 1, num_units).zero_()) for _ in range(nlayers)])

        self.dropout = LockedDropout(dropout)
        self.concat = concat
        self.nlayers = nlayers
        self.return_last = return_last

        # self.reset_parameters()

    def reset_parameters(self):
        for rnn in self.rnns:
            for name, p in rnn.named_parameters():
                if 'weight' in name:
                    p.data.normal_(std=0.1)
                else:
                    p.data.zero_()

    def get_init(self, bsz, i):
        return self.init_hidden[i].expand(-1, bsz, -1).contiguous(), self.init_c[i].expand(-1, bsz, -1).contiguous()

    def forward(self, input, input_lengths=None):
        lengths = torch.tensor(input_lengths)
        lens, indices = torch.sort(lengths, 0, True)
        input = input[indices]
        _, _indices = torch.sort(indices, 0)

        bsz, slen = input.size(0), input.size(1)
        output = input
        outputs = []
        #if input_lengths is not None:
        #    lens = input_lengths.data.cpu().numpy()
        lens[lens==0] = 1

        for i in range(self.nlayers):
            hidden, c = self.get_init(bsz, i)

            output = self.dropout(output)
            if input_lengths is not None:
                output = rnn.pack_padded_sequence(output, lens, batch_first=True)

            output, (hidden, c) = self.rnns[i](output, (hidden, c))


            if input_lengths is not None:
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
                if output.size(1) < slen: # used for parallel
                    padding = Variable(output.data.new(1, 1, 1).zero_())
                    output = torch.cat([output, padding.expand(output.size(0), slen-output.size(1), output.size(2))], dim=1)
            if self.return_last:
                outputs.append(hidden.permute(1, 0, 2).contiguous().view(bsz, -1))
            else:
                outputs.append(output)
        for i, output in enumerate(outputs):
            outputs[i] = output[_indices]
        if self.concat:
            return torch.cat(outputs, dim=-1)
        return outputs[-1]

class BiAttention(nn.Module):
    def __init__(self, input_size, dropout):
        super().__init__()
        self.dropout = LockedDropout(dropout)
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = self.dropout(input)
        memory = self.dropout(memory)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot
        att = att - 1e30 * (1 - mask[:,None])

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x shape: [batch_size, seq_length, hidden_size]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力得分
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights = self.softmax(attention_scores)

        # 应用注意力权重
        attention_output = torch.matmul(attention_weights, V)
        return attention_output
    
    
class EnhancedBERTWithAttention(nn.Module):
    def __init__(self, config):
        super(EnhancedBERTWithAttention, self).__init__()
        # BERT模型
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        bert_hidden_size = self.bert.config.hidden_size
        
        # 多头注意力机制
        self.multihead_attention = MultiHeadAttention(hidden_size=bert_hidden_size, num_heads=config.num_attention_heads)
        
        # 其他属性...
        self.linear_re = nn.Linear(bert_hidden_size, config.relation_num)
        # 初始化其他组件...
        
    def forward(self, *input):
        # 从BERT获取编码
        context_output = self.bert(*input)[0]
        
        # 应用多头注意力机制
        attention_output = self.multihead_attention(context_output, context_output, context_output)
        
        # 应用最终的线性层
        output = self.linear_re(attention_output)
        
        return output