## Copyright 2020 UT-Battelle, LLC.  See LICENSE.txt for more information.
###
 # @author Narasinga Rao Miniskar, Frank Liu, Dwaipayan Chakraborty, Jeffrey Vetter
 #         miniskarnr@ornl.gov
 # 
 # Modification:
 #              Baseline code
 # Date:        Apr, 2020
 #**************************************************************************
###

"""
  torch_wl_cnn.py: define the cnn model for gem5 data
"""

import torch.nn as nn

"""
 flatten features, dim-0 is batch size
"""
def flatten_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

class torch_wl_cnn(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self._n_inputs = n_inputs
        self._param1 = 32
        self._param2 = 64
        self._param3 = 32
        
        self._conv_blk = nn.Sequential(
            nn.Conv1d(1, self._param1, kernel_size=3, padding=1),
            nn.BatchNorm1d( self._param1 ),
            nn.Tanh(),
            ## no maxpooling
            nn.Conv1d( self._param1, self._param2, kernel_size=3, padding=1),
            nn.BatchNorm1d( self._param2 ),
            nn.Tanh()
            ## no maxpooling 
            )
        
        self._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear( self._param2*self._n_inputs, self._param3),
            nn.Tanh(),         # nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self._param3, 1),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self._conv_blk(x)
        x = x.view(-1, flatten_features(x))
        x = self._fc(x)
        return x


class torch_wl_cnn_5(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self._n_inputs = n_inputs
        self._param1 = 32
        self._param2 = 64
        self._param3 = 32
        
        self._conv_blk = nn.Sequential(
            nn.Conv1d(1, self._param1, kernel_size=5, padding=2),
            nn.BatchNorm1d( self._param1 ),
            nn.Tanh(),       ## nn.ReLU(inplace=True),
            ## no maxpooling
            nn.Conv1d( self._param1, self._param2, kernel_size=5, padding=2),
            nn.BatchNorm1d( self._param2 ),
            nn.Tanh(),       ## nn.ReLU(inplace=True)
            ## no maxpooling 
            )
        
        self._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear( self._param2*self._n_inputs, self._param3),
            nn.Tanh(),          ## nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self._param3, 1),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self._conv_blk(x)
        x = x.view(-1, flatten_features(x))
        x = self._fc(x)
        return x

    
        
class torch_wl_cnn_16(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self._n_inputs = n_inputs
        self._param1 = 32
        self._param2 = 16
        self._param3 = 16
        
        self._conv_blk = nn.Sequential(
            nn.Conv1d(1, self._param1, kernel_size=5, padding=2),
            nn.BatchNorm1d( self._param1 ),
            nn.Tanh(),       ## nn.ReLU(inplace=True),
            ## no maxpooling
            nn.Conv1d( self._param1, self._param2, kernel_size=5, padding=2),
            nn.BatchNorm1d( self._param2 ),
            nn.Tanh(),       ## nn.ReLU(inplace=True)
            ## no maxpooling 
            )
        
        self._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear( self._param2*self._n_inputs, self._param3),
            nn.Tanh(),          ## nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self._param3, 1),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self._conv_blk(x)
        x = x.view(-1, flatten_features(x))
        x = self._fc(x)
        return x

class torch_wl_cnn_64(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self._n_inputs = n_inputs
        self._param1 = 64
        self._param2 = 32
        self._param3 = 16
        
        self._conv_blk = nn.Sequential(
            nn.Conv1d(1, self._param1, kernel_size=5, padding=2),
            nn.BatchNorm1d( self._param1 ),
            nn.Tanh(),       ## nn.ReLU(inplace=True),
            ## no maxpooling
            nn.Conv1d( self._param1, self._param2, kernel_size=5, padding=2),
            nn.BatchNorm1d( self._param2 ),
            nn.Tanh(),       ## nn.ReLU(inplace=True)
            ## no maxpooling 
            )
        
        self._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear( self._param2*self._n_inputs, self._param3),
            nn.Tanh(),          ## nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self._param3, 1),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self._conv_blk(x)
        x = x.view(-1, flatten_features(x))
        x = self._fc(x)
        return x
    
"""
 More generic network, this is not the best way, but it will do for the time being
"""
class torch_wl_cnn_gen(nn.Module):
    def __init__(self, nn_topo):
        super().__init__()
        n_inputs, para1, para2, para3 = nn_topo
        self._n_inputs = n_inputs
        self._param1 = para1
        self._param2 = para2
        self._param3 = para3
        
        self._conv_blk = nn.Sequential(
            nn.Conv1d(1, self._param1, kernel_size=3, padding=1),
            nn.BatchNorm1d( self._param1 ),
            nn.Tanh(),
            ## no maxpooling
            nn.Conv1d( self._param1, self._param2, kernel_size=3, padding=1),
            nn.BatchNorm1d( self._param2 ),
            nn.Tanh()
            ## no maxpooling 
            )
        
        self._fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear( self._param2*self._n_inputs, self._param3),
            nn.Tanh(),         # nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self._param3, 1),
            nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self._conv_blk(x)
        x = x.view(-1, flatten_features(x))
        x = self._fc(x)
        return x
