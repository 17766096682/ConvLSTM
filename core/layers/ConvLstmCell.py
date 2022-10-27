import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, configs):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = configs.filter_size
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        self.conv = nn.Conv2d(in_channels=self.input_channels + self.hidden_dim,   #因为是cat的所以需要相加
                              out_channels=4 * self.hidden_dim,               #
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=True)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) #输出被乘以了4 这里被拆解了
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next,(h_next, c_next)


    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
if __name__ == '__main__':
    from configs.radar_train_configs import configs


    parse = configs()
    configs = parse.parse_args()
    print(configs.num_hidden)

    model = ConvLSTMCell(configs.num_hidden, configs.num_hidden, configs).cuda()
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))


    # a = torch.randn((5,1,28,28))
    # h = torch.zeros((5,32,28,28))
    # c = h
    # h,c = model(a,(h,c))
    # print(a.shape)
