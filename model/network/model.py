import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils import *
from .base_network import *
from transformers import AutoTokenizer, AutoModel


class Encoder(nn.Module):
    """ Bert/BiLSTM Encoder Model."""
    def __init__(self, encoder_type, encoder_param, config=None, corpus=None):
        """Bi-LSTM Encoder

        Args:
            encoder_type: (str) BERT/LSTM
            encoder_param: (Dict) Encoder Hyperparameters
            config: (Dict) configuration dict for BiLSTM
            corpus: (Object) Corpus Class for BiLSTM
        """
        super(Encoder, self).__init__()

        # init
        self.encoder_type = encoder_type
        if self.encoder_type == 'lstm':
            self.vocab_size = config['dataset']['vocab']['vocab size']
            self.input_size = config['dataset']['vocab']['word2vec dim']
            self.hidden_size = encoder_param['hidden_size']
            self.num_layers = encoder_param['num_layers']
            self.bi = encoder_param['bi']
            self.freeze_emb = encoder_param['freeze_emb']
            if config['dataset']['pretrain']:
                wv_tensor = corpus.get_wordembedding()
                self.embedding = nn.Embedding.from_pretrained(wv_tensor, freeze=self.freeze_emb, padding_idx=0)
            else:
                self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                              embedding_dim=self.input_size,
                                              padding_idx=0)
            self.bilstm = BiLSTM(self.input_size,
                                 self.hidden_size,
                                 self.num_layers,
                                 self.bi)

        elif self.encoder_type == 'bert':
            bert_model_name = encoder_param.get('bert_model_name', 'bert-base-uncased')
            self.bert = AutoModel.from_pretrained(bert_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
            self.bert_size = self.bert.embeddings.word_embeddings.embedding_dim

        if 'tau' in encoder_param:
            self.tau = encoder_param['tau']

    def bert_forward(self, *input):
        x, x_len, return_type = input  # (batch_size, max_len, word2vec_dim) (batch_size, )
        batch_input = self.tokenizer(x, return_tensors="pt", padding=True)
        batch_input = {k: v.to(self.bert.device) for k, v in batch_input.items()}
        output = self.bert(**batch_input)

        if return_type == 'mean_pooling':
            output = output.last_hidden_state[:, 1:-1, :].sum(dim=1).div(x_len.float().unsqueeze(-1))
        elif return_type == 'all_return':
            output = output.last_hidden_state[:, 1:-1, :]
        return output

    def lstm_forward(self, *input):
        x, x_len, return_type = input  # (batch_size, max_len, word2vec_dim) (batch_size, )
        # Embed
        x_emb = self.embedding(x)
        # BiLSTM
        out = self.bilstm(x_emb, x_len, return_type)
        return out

    def forward(self, *input):
        x, x_len, return_type = input
        if return_type == 'encode':
            return_type = 'mean_pooling'
        if self.encoder_type == 'lstm':
            return self.lstm_forward(x, x_len, return_type)
        elif self.encoder_type == 'bert':
            return self.bert_forward(x, x_len, return_type)


class Extractor(nn.Module):
    """Semantic Extractor Network. The true purpose is to
    generate new parameters for the adaptive representations of the encoder."""
    def __init__(self, emb_size, d_r):
        super(Extractor, self).__init__()

        self.emb_size = emb_size
        # self.alpha = alpha
        # self.W1 = nn.Linear(emb_size, emb_size, bias=False)
        self.W1 = nn.Parameter(torch.rand(emb_size, emb_size))
        # self.W2 = nn.Linear(64, emb_size, bias=False)
        self.W2 = nn.Parameter(torch.rand(emb_size, emb_size))
        self.W3 = nn.Parameter(torch.rand(d_r, emb_size))
        self._reset_parmaters()
        # self.decoder = nn.TransformerDecoderLayer(d_model=emb_size, nhead=4)

    def _reset_parmaters(self):
        for name, param in self.named_parameters():
            torch.nn.init.kaiming_normal_(param)

    def forward(self, *input):
        residual = input[0]
        # memory_protos, novel_protos, after_memory_protos, after_novel_protos = input
        torch.autograd.set_detect_anomaly(True)

        # after_protos = torch.cat([after_memory_protos, after_novel_protos], 0)
        # protos = torch.cat([memory_protos, novel_protos], 0)
        # delta = after_protos - protos
        matrix_F = residual.matmul(self.W1)
        matrix_A = F.softmax(self.W3.matmul(torch.relu(self.W2.matmul(matrix_F.t()))), dim=-1)
        semantic_components = matrix_A.matmul(matrix_F)
        return semantic_components


class SampleAdaptNet(nn.Module):
    """Sample adaptation network using semantic components to calibrate attention scores."""
    def __init__(self, emb_size):
        super(SampleAdaptNet, self).__init__()
        self.emb_size = emb_size
        # self._reset_parmaters()

    # def _reset_parmaters(self):
    #     for name, param in self.named_parameters():
    #         torch.nn.init.kaiming_normal_(param)

    def forward(self, *input):
        querys_w, querys_len, semantic_components, beta, alpha = input

        bs, seq_len, dim = querys_w.size(0), querys_w.size(1), querys_w.size(2)

        # TODO: alpha 10
        beta = beta * alpha
        # beta=10

        score, idx = cos_sim(querys_w.reshape(-1, self.emb_size), semantic_components).reshape(bs, seq_len, -1).max(-1)
        attn_score = score * beta
        attn_score.masked_fill_(attn_score == 0, -(1e+9))
        attn = F.softmax(attn_score, dim=-1)
        # print(attn)
        out = querys_w.transpose(1, 2).matmul(attn.unsqueeze(-1)).squeeze(-1)

        return out


class LTA(nn.Module):
    """Learn to Adapt (LTA) main model."""
    def __init__(self, config, encoder_type, encoder_param, n_seen_class, tau, alpha, d_r, corpus=None):
        """

        Args:
            config: (Object) configuration
            encoder_type: (str) lstm or bert
            encoder_param: (Dict) parameters of the encoder
            n_seen_class: (int) the number of seen classes
            tau: (float) initial temperature scalar
            alpha: (float) initial temperature scalar
            d_r: (float) semantic component heads
            corpus: (Object) If encoder is lstm
        """

        super(LTA, self).__init__()

        # init
        self.config = config
        self.encoder_type = encoder_type
        self.encoder_param = encoder_param

        ## embedding
        # self.vocab_size = config['dataset']['vocab']['vocab size']
        # self.input_size = config['dataset']['vocab']['word2vec dim']
        # self.freeze_emb = freeze_emb
        ## BiLSTM
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        # self.dropout = dropout
        # self.bi = bi

        self.num_classes = n_seen_class

        self.tau = nn.Parameter(torch.tensor(tau))
        self.alpha = alpha
        self.d_r = d_r

        # encoder
        if self.encoder_type == 'lstm':
            self.encoder = Encoder(encoder_type, encoder_param, config, corpus)
            self.emb_size = encoder_param['hidden_size'] * 2 if encoder_param['bi'] else encoder_param['hidden_size']
        elif self.encoder_type == 'bert':
            self.encoder = Encoder(encoder_type, encoder_param)
            self.bert_size = self.encoder.bert.embeddings.word_embeddings.embedding_dim
            self.emb_size = self.bert_size

        # matrix S
        self.seen_class_proto_list = [nn.Parameter(torch.randn(self.emb_size)) for _ in range(self.num_classes)]
        self.seen_class_protos = nn.ParameterList(self.seen_class_proto_list)
        self._reset_parameters()

        if self.config['arch_step2']['ablation']['proto_adapt']:
            # self.self_attn_layer = nn.MultiheadAttention(embed_dim=self.emb_size, num_heads=4)
            self_attn_layer = nn.TransformerEncoderLayer(d_model=self.emb_size, nhead=4, dim_feedforward=2048)
            self.proto_adapt_net = nn.TransformerEncoder(self_attn_layer, num_layers=1)
        if self.config['arch_step2']['ablation']['feat_adapt']:
            self.extractor = Extractor(self.emb_size, d_r)  # semantic extractor
            self.sample_adapt_net = SampleAdaptNet(self.emb_size)  # word-level attention calibration
            #TODO alpha

    @torch.no_grad()
    def _reset_parameters(self):
        for i, param in enumerate(self.seen_class_protos):
            nn.init.kaiming_normal_(param.unsqueeze_(0)).squeeze_()

    def forward(self, *input):
        type = input[-1]
        if type == 'encode':
            x, x_len = input[:-1]
            out = self.encoder(x, x_len, type)
            return out

        elif type == 'proto_adapt':
            unseen_protos, seen_y = input[:-1]

            device = unseen_protos.device
            before_protos = torch.zeros(len(self.seen_class_protos), self.emb_size).to(device)
            for i, x in enumerate(self.seen_class_protos):
                before_protos[i] = self.seen_class_protos[i]  # matrix S
            seen_protos = before_protos[seen_y]  # matrix S_i

            protos = torch.cat([seen_protos, unseen_protos], 0)  # matrix R

            semantic_components = None
            if self.config['arch_step2']['ablation']['proto_adapt']:
                # src2, score = self.proto_adapt_net(protos.unsqueeze(1), protos.unsqueeze(1), protos.unsqueeze(1))  # if use self attention only
                residual = self.proto_adapt_net(protos.unsqueeze(1)).squeeze(1)  # matrix Z

                protos = protos + residual # matrix R hat
                # after_seen_protos = after_protos[:len(seen_protos)]
                # after_unseen_protos = after_protos[len(seen_protos):]

                if self.config['arch_step2']['ablation']['feat_adapt']:
                    semantic_components = self.extractor(residual)
            return protos, semantic_components
                    # return before_protos, after_protos, semantic_components
                    # v, loss_r = self.generator(memory_protos, novel_protos, after_memory_protos, after_novel_protos)
                    # return after_protos, memory_protos, novel_protos, after_memory_protos, after_novel_protos, v, loss_r
                # else:
                #     return before_protos, after_protos, None
                    # return after_protos, memory_protos, novel_protos, after_memory_protos, after_novel_protos, 0, 0
            # else:
                # src = protos
                # after_protos = src
                # after_memory_protos = after_protos[:len(memory_protos)]
                # after_novel_protos = after_protos[len(memory_protos):]

                # If sample adaptation is needed, it should output semantic components using extractor.
                # components = None
                # if self.config['arch_step2']['ablation']['feat_adapt']:
                #     components = self.extractor(protos)
                # #     return self.
                #     v, loss_r = self.generator(memory_protos, novel_protos, after_memory_protos, after_novel_protos)
                #     return after_protos, memory_protos, novel_protos, after_memory_protos, after_novel_protos, v, loss_r
                # return protos, components
                # return after_protos, memory_protos, novel_protos, after_memory_protos, after_novel_protos, 0, 0

        elif type == 'sample_adapt':
            querys_x, querys_len, protos, semantic_components, seen_len = input[:-1]
            # querys_x, querys_len, memory_protos, novel_protos, after_memory_protos, after_novel_protos, v = input[:-1]

            if self.config['arch_step2']['ablation']['feat_adapt']:
                querys_w = self.encoder(querys_x, querys_len, 'all_return')
                after_seen_protos = protos[:seen_len]
                after_unseen_protos = protos[seen_len:]
                unseen_len = len(protos) - seen_len
                beta = cos_mmd(after_seen_protos, after_unseen_protos)  # beta is for continual testing (Section 4.5)
                beta = beta * (seen_len + unseen_len) / seen_len
                querys = self.sample_adapt_net(querys_w, querys_len, semantic_components, beta, self.alpha)
                return querys
            else:
                querys = self.encoder(querys_x, querys_len, 'mean_pooling')
                return querys
