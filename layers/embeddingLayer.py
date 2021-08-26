from torch import FloatTensor
from torch import squeeze
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingLayer(nn.Module):
    def __init__(self, num_classes, in_feature, out_features, kernel_size=5, device="cpu"):
        """EmbeddingLayer

        Args:
            num_classes ([type]): 字典的宽度
            kernel_size ([type]): 一维卷积核的大小
            in_feature ([type]): 传入的features
            out_features ([type]): 传出的features
            device ([type]): device
        """
        super(EmbeddingLayer, self).__init__()

        self.device = device
        self.num_classes = num_classes

        kernel_size_2 = in_feature - 2 * (kernel_size - 1)
        self.embedding_model = nn.Sequential(
            nn.Conv1d(num_classes,
                      16,
                      kernel_size=kernel_size
                      ),  # Shape: [U, 16, C - (char_conv_l1_kernel_size - 1)]
            nn.LeakyReLU(),
            nn.MaxPool1d(
                kernel_size=kernel_size, stride=1
            ),  # Shape: [U, 16, C - 2*(char_conv_l1_kernel_size - 1)]
            nn.Conv1d(
                16,
                int(out_features),
                kernel_size=kernel_size_2),  # Shape: [U, 1, D]
            nn.LeakyReLU(),
        )
    
    def forward(self, initial_embedding):
        initial_embedding_one_hot = F.one_hot(initial_embedding, num_classes=self.num_classes)
        initial_embedding_one_hot = initial_embedding_one_hot.type(FloatTensor).to(self.device)
        initial_embedding_one_hot_permute = initial_embedding_one_hot.permute(0, 2, 1)  # shape: U, A, C  

        embedding = self.embedding_model(initial_embedding_one_hot_permute)
        embedding_sq = squeeze(embedding, dim=-1)

        return embedding_sq


if __name__ == '__main__':
    import torch
    a = torch.randint(low=0, high=70, size=(10, 16))
    print(a)
    model = EmbeddingLayer(num_classes=70, in_feature=16, out_features=25, device="cpu", kernel_size=5)
    b = model(a)
    print(b)
    print("")