import torch
import torch.nn as nn
from models.operations import *
from models import genotypes as gt
import math

BN_MOMENTUM = 0.1


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
    
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)
        

class Upsample(nn.Module):

    def __init__(self, upsample, upsample_concat, C_prev_prev, C_prev):
        super(Upsample, self).__init__()

        self.preprocess0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            ReLUConvBN(C_prev_prev, C_prev//4, 1, 1, 0, affine=True)
        )
        self.preprocess1 = ReLUConvBN(C_prev, C_prev//4, 1, 1, 0, affine=True)
    
        op_names, indices = zip(*upsample)
        concat = upsample_concat
        self._compile(C_prev//4, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
    
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2*i]]
            h2 = states[self._indices[2*i+1]]
            op1 = self._ops[2*i]
            op2 = self._ops[2*i+1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)
        
        
    
class Network(nn.Module):

    def __init__(self, cfg, genotype):
        super(Network, self).__init__()
        
        self._num_classes = cfg.MODEL.NUM_JOINTS
        self._layers = cfg.TRAIN.LAYERS
        self.C = cfg.TRAIN.INIT_CHANNELS
        self.genotype = genotype
        
        assert self._layers == 12 or self._layers == 18, \
            'ERROR: we only support 12 layers or 18 layers right now'
        
        if self._layers == 12:
            self.cell_layer = [3,6,9]
            self.output_layer = [2,5,8,11]
        elif self._layers == 18:
            self.cell_layer = [3,8,15]
            self.output_layer = [2,7,14,17]
            
        self.stem0 = nn.Sequential(
            nn.Conv2d(3, self.C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.C, momentum=BN_MOMENTUM),
            nn.ReLU()
        )

        self.stem1 = nn.Sequential(
            nn.Conv2d(self.C, self.C*2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.C*2, momentum=BN_MOMENTUM),
            nn.ReLU()
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(self.C*2, self.C*2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.C*2, momentum=BN_MOMENTUM)
        )
        self.relu = nn.ReLU()
    
        C_prev_prev, C_prev, C_curr = self.C*2, self.C*2, int(self.C/2)
        self.cells = nn.ModuleList()
        self.num_inchannels = []
        reduction_prev = False
        for i in range(self._layers):
            if i in self.output_layer:
                self.num_inchannels.append(int(C_curr*4))
                
            if i in self.cell_layer:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(gt.BACKBONE, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

        self.num_inchannels = self.num_inchannels[::-1]
        
        self.upsamples = nn.ModuleList()
        upsample = Upsample(genotype.upsample1, genotype.upsample_concat1, self.num_inchannels[0], self.num_inchannels[1])
        self.upsamples += [upsample]
        upsample = Upsample(genotype.upsample2, genotype.upsample_concat2, self.num_inchannels[1], self.num_inchannels[2])
        self.upsamples += [upsample]
        upsample = Upsample(genotype.upsample3, genotype.upsample_concat3, self.num_inchannels[2], self.num_inchannels[3])
        self.upsamples += [upsample]
            

        self.final_layer = nn.Conv2d(
            in_channels=self.num_inchannels[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    def forward(self,x):
    
        s0 = self.stem0(x)
        s0 = self.stem1(s0)
        s1 = self.stem2(s0)
        features = []
        
        for i, cell in enumerate(self.cells):

            s0, s1 = s1, cell(s0, s1)
            
            if i in self.output_layer:
                features.append(s1)
                
        out = self.upsamples[0](features[3],features[2])      
        out = self.upsamples[1](out,features[1])
        out = self.upsamples[2](out,features[0])
        
        out = self.relu(out)
        logits = self.final_layer(out)
        
        return logits


    def init_weights(self, pretrained=''):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)  


    


