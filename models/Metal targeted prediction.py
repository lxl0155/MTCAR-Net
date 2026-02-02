import torch
import torch.nn as nn


class CAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CAM, self).__init__()

        # 1维池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Sequential(nn.Linear(in_planes, in_planes // ratio, bias=False),
                                nn.ReLU(),
                                nn.Linear(in_planes // ratio, in_planes, bias=False))
        self.fc2 = nn.Sequential(nn.Linear(in_planes, in_planes // ratio, bias=False),
                                nn.ReLU(),
                                nn.Linear(in_planes // ratio, in_planes, bias=False))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
     
        x = x.permute(0,2,1)

        avg_out = self.fc1(self.avg_pool(x).permute(0,2,1))
        max_out = self.fc2(self.max_pool(x).permute(0,2,1))

        out = avg_out + max_out
        return self.sigmoid(out)
    

class ChannelAttentionLayer(nn.Module):
    def __init__(self, in_planes, ratio=16, mlp_hidden_dim=64, cls = 1):
        super(ChannelAttentionLayer, self).__init__()
        self.cls = cls
        self.cam = CAM(in_planes, ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1)  
        )
    
    def forward(self, x):
        batch_size, _, dim = x.size()
        cls = self.cls

        out = []
        for i in range(cls):
            # 通过 CAM 模块处理
            cam_output = self.cam(x)   
            cam_output = x * cam_output
            
            cam_output = cam_output[:,i,:].unsqueeze(1)

            # 通过 MLP 进一步处理 CAM 输出
            mlp_output = self.mlp(cam_output.view(batch_size, dim))  
            
            # 将 MLP 的输出作为最终结果
            out.append(mlp_output)  

        # 将处理后的所有输出拼接在一起，得到最终的输出
        out = torch.cat(out, dim=1)  
        
        return out