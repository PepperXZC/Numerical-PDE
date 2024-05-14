from __future__ import print_function

import torch
from pyDOE import lhs
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

np.random.seed(1234)

print("Cuda availability: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 20), nn.Tanh(),

            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),

            nn.Linear(20, 2))                    
        
    def forward(self, x):
        return self.net(x)
    
def function(self, t, x, y):
    nu = 0.01   # Re = 100

    res = self.net(torch.hstack((t, x, y)))
    psi, p = res[:, 0:1], res[:, 1:2]    # NN output: velocity field and pressure field 

    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0].to(device)
    v = -1.*torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0].to(device)

    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0].to(device)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0].to(device)
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0].to(device)
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0].to(device)

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0].to(device)
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0].to(device)
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0].to(device)
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0].to(device)

    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0].to(device)
    p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0].to(device)

    f = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    g = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    grad = torch.hstack((u_x, u_y, v_x, v_y))
    return u, v, p, f, g, grad

# time
T = 50
nt = 50
t = np.linspace(0, T, nt)

##### Boundary points #####
boundary_num = 50
t_bound = t.repeat(boundary_num)
t_bound = Variable(torch.from_numpy(np.reshape(t_bound, (-1, 1))).float(), requires_grad = True).to(device)

# upper boundary
x_up = np.tile(np.linspace(0., 1., boundary_num), nt)
y_up = np.ones(boundary_num*nt)
x_up_in = Variable(torch.from_numpy(np.reshape(x_up, (-1, 1))).float(), requires_grad = True).to(device)
y_up_in = Variable(torch.from_numpy(np.reshape(y_up, (-1, 1))).float(), requires_grad = True).to(device)   

# lower boundary
x_low = np.tile(np.linspace(0., 1., boundary_num), nt)
y_low = np.zeros(boundary_num*nt)
x_low_in = Variable(torch.from_numpy(np.reshape(x_low, (-1, 1))).float(), requires_grad = True).to(device)
y_low_in = Variable(torch.from_numpy(np.reshape(y_low, (-1, 1))).float(), requires_grad = True).to(device)      

# left boundary
x_left = np.zeros(boundary_num*nt)
y_left = np.tile(np.linspace(0., 1., boundary_num), nt)
x_left_in = Variable(torch.from_numpy(np.reshape(x_left, (-1, 1))).float(), requires_grad = True).to(device)
y_left_in = Variable(torch.from_numpy(np.reshape(y_left, (-1, 1))).float(), requires_grad = True).to(device)   

# right boundary
x_right = np.ones(boundary_num*nt) 
y_right = np.tile(np.linspace(0., 1., boundary_num), nt)
x_right_in = Variable(torch.from_numpy(np.reshape(x_right, (-1, 1))).float(), requires_grad = True).to(device)
y_right_in = Variable(torch.from_numpy(np.reshape(y_right, (-1, 1))).float(), requires_grad = True).to(device)  


##### Residual points #####
res_num = 5000
res_pts = lhs(2, res_num)

t_res = t.repeat(res_num)
t_res = Variable(torch.from_numpy(np.reshape(t_res, (-1, 1))).float(), requires_grad = True).to(device)
x_res = np.tile(res_pts[:, 0], nt)
y_res = np.tile(res_pts[:, 1], nt)
x_res_in = Variable(torch.from_numpy(np.reshape(x_res, (-1, 1))).float(), requires_grad = True).to(device)
y_res_in = Variable(torch.from_numpy(np.reshape(y_res, (-1, 1))).float(), requires_grad = True).to(device)

######### t=0 ##########
x_0 = np.concatenate((x_up[:boundary_num], x_low[:boundary_num], x_left[:boundary_num], x_right[:boundary_num], x_res[:res_num]), axis=0)
y_0 = np.concatenate((y_up[:boundary_num], y_low[:boundary_num], y_left[:boundary_num], y_right[:boundary_num], y_res[:res_num]), axis=0)
t_0 = np.zeros(boundary_num*4 + res_num)
t_origin = Variable(torch.from_numpy(np.reshape(t_0, (-1, 1))).float(), requires_grad = True).to(device)
x_origin_in = Variable(torch.from_numpy(np.reshape(x_0, (-1, 1))).float(), requires_grad = True).to(device)
y_origin_in = Variable(torch.from_numpy(np.reshape(y_0, (-1, 1))).float(), requires_grad = True).to(device) 


net = Net()
print(net)
net = net.to(device)

##### Xavier Initialization #####
for m in net.modules():
    if isinstance(m, (nn.Linear)):
        torch.nn.init.xavier_normal_(m.weight)


##### Loss Function #####
mse_loss_func = torch.nn.MSELoss()
optimizer_adam = torch.optim.Adam(net.parameters(), lr=0.001)
loss_Adam = []

optimizer_LBFGS = torch.optim.LBFGS(net.parameters(), history_size=8, max_iter=500000)
loss_LBFGS = []


######### Training ########

#### Adam ####
N_iter_Adam = 10001
boundary_num0 = 50
res_num0 = 5000
print("------ IN Adam ------")
for n in range(N_iter_Adam):
    def closure():
        optimizer_adam.zero_grad()
        ##### t=0 #####
        u_origin, v_origin, p_origin, f_origin, g_origin, grad_origin = function(net, t_origin, x_origin_in, y_origin_in)
        zero_1 = torch.zeros(boundary_num0*4 + res_num0, 1).to(device)
        u_loss_origin = mse_loss_func(u_origin, zero_1)
        v_loss_origin = mse_loss_func(v_origin, zero_1)
        p_loss_origin = mse_loss_func(p_origin, zero_1)
        loss_origin = u_loss_origin + v_loss_origin + p_loss_origin

        ##### Boundary #####
        boundary_num = x_up_in.shape[0]
        res_num = x_res_in.shape[0]
        zeros = torch.zeros(boundary_num, 1).to(device)
        ones = torch.ones(boundary_num, 1).to(device)

        # upper boundary 
        u_up, v_up, p_up, f_up, g_up, grad_up = function(net, t_bound, x_up_in, y_up_in)
        u_loss_up = mse_loss_func(u_up, ones)           # u = 1 at upper boundary 
        v_loss_up = mse_loss_func(v_up, zeros)          # v = 0 at upper boundary 
        loss_up = u_loss_up + v_loss_up

        # lower boundary 
        u_low, v_low, p_low, f_low, g_low, grad_right = function(net,t_bound, x_low_in, y_low_in)
        u_loss_low = mse_loss_func(u_low, zeros)        # u = 0 at lower boundary 
        v_loss_low = mse_loss_func(v_low, zeros)        # v = 0 at lower boundary 
        loss_low = u_loss_low + v_loss_low

        # left boundary 
        u_left, v_left, p_left, f_left, g_left, grad_left = function(net,t_bound, x_left_in, y_left_in)
        u_loss_left = mse_loss_func(u_left, zeros)      # u = 0 at lower boundary
        v_loss_left = mse_loss_func(v_left, zeros)      # v = 0 at lower boundary
        loss_left = u_loss_left + v_loss_left

        # right boundary 
        u_right, v_right, p_right, f_right, g_right, grad_right = function(net,t_bound, x_right_in, y_right_in)
        u_loss_right = mse_loss_func(u_right, zeros)        # u = 0 at lower boundary
        v_loss_right = mse_loss_func(v_right, zeros)        # v = 0 at lower boundary
        loss_right = u_loss_right + v_loss_right

        ##### Residual #####
        res_num = x_res_in.shape[0]
        zeros1 = torch.zeros(res_num, 1).to(device)

        u_res, v_res, p_res, f_res, g_res, grad_res = function(net,t_res, x_res_in, y_res_in)
        f_loss_res = mse_loss_func(f_res, zeros1)       # to satisfy the PDE
        g_loss_res = mse_loss_func(g_res, zeros1)       # to satisfy the PDE
        grad_loss_origin = mse_loss_func(grad_res, zeros1)
        loss_res = f_loss_res + g_loss_res

        # LOSS FUNCTION:
        loss = loss_origin + loss_up + loss_low + loss_left + loss_right + loss_res 
        loss.backward()
        return loss
       
    optimizer_adam.step(closure)
    loss = closure() 
    loss_Adam.append(loss.cpu().detach().numpy())
    
    # if n%100 == 0:
    print("Adam - Epoch: ", n, "Training Loss: ", loss.item())

