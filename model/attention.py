import torch 
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision

#Spatial Guided Attention 
class SpatialGuidedAttn(nn.Module):
    def __init__(self, in_dim):
        super(SpatialGuidedAttn, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_training, x_pre):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x_training.size()
        proj_query = self.query_conv(x_pre).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x_training).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x_training).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        attn = self.softmax(out)
        
        #out = self.gamma*out + x_pre       
        #return out
        
        return attn
    
    
    
#Channel Guided Attention
class ChannelGuidedAttn(nn.Module):
    def __init__(self, in_dim):
        super(ChannelGuidedAttn, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x_training, x_pre):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x_training.size()
        proj_query = x_pre.view(m_batchsize, C, -1)
        proj_key = x_training.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x_training.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        
        attn = self.softmax(out)

        #out = self.gamma*out + x_pre        
        #return out
        
        return attn 

    
    