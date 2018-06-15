import torch
import torch.nn as nn
import numpy as np
import config
import decoder

class Model(decoder.Ctx2SeqAttention):
    def __init__(self, vocab_size):
        super(Model, self).__init__(
            ctx_dim = config.encoder_hidden_dim,
            num_steps = config.max_question_len,
            vocab_size = vocab_size,
            src_hidden_dim = config.dense_vector_dim,
            trg_hidden_dim = config.dense_vector_dim//2,
            pad_token = config.NULL_ID,
            bidirectional = True,
            nlayers = config.num_question_encoder_layers,
            nlayers_trg = config.num_decoder_rnn_layers,
            dropout=0)
        self.embedding = torch.nn.Embedding(self.vocab_size, config.embedding_dim)
        self.passage_conv0 = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size0, config.conv_vector_dim)
        self.passage_conv1 = self.cnn_layers(config.num_passage_encoder_layers, config.encoder_kernel_size1, config.conv_vector_dim)
        self.encoder_dim = config.encoder_hidden_dim * 2
        self.passage_dense = nn.Linear(self.encoder_dim, config.dense_match_dim)
        self.passage_encoder = nn.LSTM(config.conv_vector_dim*2, config.encoder_hidden_dim, 2, bidirectional=True, batch_first=True)
        self.state_dense = nn.Linear(config.dense_vector_dim*4, config.dense_vector_dim)


    def forward(self, x, y):
        ctx, state_x, ctx_mask = self.encode_passage(x)
        batch_size = y.shape[0]
        num_questions = y.shape[1]
        sos = torch.LongTensor(batch_size, num_questions, 1).fill_(config.SOS_ID).cuda()
        y = torch.cat([sos, y], 2)
        question_embed = self.embedding(y)
        decoder_logit = super(Model, self).forward(ctx, state_x, ctx_mask, question_embed)
        return decoder_logit


    def encode_passage(self, input):
        embed = self.embedding(input)
        encoding0 = self.encode_embedding(self.passage_conv0, embed)
        encoding1 = self.encode_embedding(self.passage_conv1, embed)
        encoding = torch.cat([encoding0, encoding1], -1)
        dense = self.passage_dense(encoding)
        weight_dim = dense.shape[-1]
        coref = torch.bmm(dense, dense.transpose(1, 2)) / (weight_dim**0.5)
        mask = (input != 0).float()
        coref -= (1-mask.unsqueeze(1)) * 100000
        alpha = nn.functional.softmax(coref, -1)
        context = torch.bmm(alpha, encoding)
        ctx, (state_h, state_c) = self.passage_encoder(context)
        state = torch.cat([state_h.transpose(0, 1), state_c.transpose(0, 1)], -1).view(embed.shape[0], -1)
        state = self.state_dense(state)
        return ctx, state, mask
        

    def selfmatch(self, x, dense0, dense1, mask=None):
        coref = torch.bmm(dense0(x), dense1(x).transpose(1, 2))
        if mask is not None:
            coref = coref - (1-mask)*10000
        alpha = torch.sigmoid(coref)
        return torch.bmm(alpha, x)


    def encode_embedding(self, convs, embed):
        x = torch.transpose(embed, 1, 2)
        x = convs(x)
        encoding = torch.transpose(x, 1, 2)
        return encoding


    def cnn_layers(self, num_layers, kernel_size, out_channels):
        modules = nn.Sequential()
        for i in range(num_layers):
            conv = nn.Conv1d(config.embedding_dim if i == 0 else out_channels, out_channels, kernel_size, padding=kernel_size//2)
            modules.add_module('conv_{}'.format(i), conv)
            modules.add_module('tanh_{}'.format(i), nn.Tanh())
        return modules


    def attention_layer(self):
        dense0 = nn.Linear(config.encoder_hidden_dim*2, config.attention_weight_dim, False)
        dense1 = nn.Linear(config.encoder_hidden_dim*2, config.attention_weight_dim, False)
        return dense0, dense1


if __name__ == '__main__':
    criterion = nn.BCEWithLogitsLoss()
    model = Model(config.char_vocab_size, None)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    seq = torch.LongTensor([[1,2,3,4,5], [4,5,6,7,8]])
    target = torch.Tensor([0, 1])
    for _ in range(10):
        similarity = model(seq, seq)
        loss = criterion(similarity, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.tolist())

