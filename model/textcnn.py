import torch
import torch.nn as nn

class TextCNN(nn.Module):
    def __init__(self,
                 class_num,
                 hidden_dim=192,
                 num_channels=100,
                 kernel_size=[3, 4, 5],
                 max_len=200,
                 dropout=0.8,
                 padding_idx=0,
                 vocab_size=32000):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx)

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[0] + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[1]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[1] + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=hidden_dim, out_channels=num_channels, kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.MaxPool1d(max_len - kernel_size[2] + 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_size), class_num)

    def forward(self, x, return_feat=False):
        if type(x) == type([]):
            text, _ = x
        else:
            text = x

        embedded_sent = self.embedding(text).permute(0, 2, 1)

        conv_out1 = self.conv1(embedded_sent).squeeze(2)
        conv_out2 = self.conv2(embedded_sent).squeeze(2)
        conv_out3 = self.conv3(embedded_sent).squeeze(2)

        all_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
        final_feature_map = self.dropout(all_out)
        out = self.fc(final_feature_map)

        if return_feat:
            return out, final_feature_map

        return out

def textcnn_agnews(args):
    return TextCNN(class_num=args.class_num)