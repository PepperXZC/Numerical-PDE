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
# device = torch.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.Tanh(),

            nn.Linear(128, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 128), nn.Tanh(),

            nn.Linear(128, 2))                    
        
    def forward(self, x):
        return self.net(x)
    
    
def function(self, t, x, y):
    nu = 1 / 100   # Re = 100

    res = self(torch.hstack((t, x, y)))
    # print(res.shape, torch.hstack((t, x, y)).shape)
    psi, p = res[:, 0:1], res[:, 1:2]    # NN output: velocity field and pressure field 
# torch.Size([5200, 2]) torch.Size([5200, 3])
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
nt = 501
tlist = np.linspace(0, T, nt)

##### Boundary points #####
boundary_num = 50

# upper boundary
# x_up = np.tile(np.linspace(0., 1., boundary_num), nt)
x_up = np.linspace(0., 1., boundary_num)
y_up = np.ones(boundary_num)
x_up_in = Variable(torch.from_numpy(np.reshape(x_up, (-1, 1))).float(), requires_grad = True).to(device)
y_up_in = Variable(torch.from_numpy(np.reshape(y_up, (-1, 1))).float(), requires_grad = True).to(device)   

# lower boundary
x_low = np.linspace(0., 1., boundary_num)
y_low = np.zeros(boundary_num)
x_low_in = Variable(torch.from_numpy(np.reshape(x_low, (-1, 1))).float(), requires_grad = True).to(device)
y_low_in = Variable(torch.from_numpy(np.reshape(y_low, (-1, 1))).float(), requires_grad = True).to(device)      

# left boundary
x_left = np.zeros(boundary_num)
y_left = np.linspace(0., 1., boundary_num)
x_left_in = Variable(torch.from_numpy(np.reshape(x_left, (-1, 1))).float(), requires_grad = True).to(device)
y_left_in = Variable(torch.from_numpy(np.reshape(y_left, (-1, 1))).float(), requires_grad = True).to(device)   

# right boundary
x_right = np.ones(boundary_num) 
y_right = np.linspace(0., 1., boundary_num)
x_right_in = Variable(torch.from_numpy(np.reshape(x_right, (-1, 1))).float(), requires_grad = True).to(device)
y_right_in = Variable(torch.from_numpy(np.reshape(y_right, (-1, 1))).float(), requires_grad = True).to(device)  


##### Residual points #####
res_num = 5000
res_pts = lhs(2, res_num)

# t_res = Variable(torch.from_numpy(np.reshape(t_res, (-1, 1))).float(), requires_grad = True).to(device)
# x_res = np.tile(res_pts[:, 0], nt)
# y_res = np.tile(res_pts[:, 1], nt)
x_res = res_pts[:, 0]
y_res = res_pts[:, 1]
x_res_in = Variable(torch.from_numpy(np.reshape(x_res, (-1, 1))).float(), requires_grad = True).to(device)
y_res_in = Variable(torch.from_numpy(np.reshape(y_res, (-1, 1))).float(), requires_grad = True).to(device)

######### t=0 ##########
# x_0 = np.concatenate((x_up[:boundary_num].copy(), x_low[:boundary_num].copy(), x_left[:boundary_num].copy(), x_right[:boundary_num].copy(), x_res[:res_num].copy()), axis=0)
# y_0 = np.concatenate((y_up[:boundary_num].copy(), y_low[:boundary_num].copy(), y_left[:boundary_num].copy(), y_right[:boundary_num].copy(), y_res[:res_num].copy()), axis=0)
# t_0 = np.zeros(boundary_num*4 + res_num)
x_0 = x_res[:res_num].copy()
y_0 = y_res[:res_num].copy()
t_0 = np.zeros(res_num)


net = Net()
print(net)
net = net.to(device)
##### Xavier Initialization #####
for m in net.modules():
    if isinstance(m, (nn.Linear)):
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)


##### Loss Function #####
mse_loss_func = torch.nn.MSELoss()
# optimizer_adam = torch.optim.Adam(net.parameters(), lr=0.001)
optimizer_LBFGS = torch.optim.LBFGS(net.parameters(), history_size=8, max_iter=500000)
loss_Adam = []

# optimizer_LBFGS = torch.optim.LBFGS(net.parameters(), history_size=8, max_iter=500000)
loss_LBFGS = []
pth_path = 'model1002.pt'

######### Training ########

N_plt = 50
xlist = np.linspace(0., 1., N_plt)
ylist = np.linspace(0., 1., N_plt)
t = np.linspace(0, T, nt).repeat(N_plt)

# Tgrid, X, Y= np.meshgrid(t, xlist, ylist)
X, Y = np.meshgrid(xlist, ylist)
Xgrid = X.flatten()
Ygrid = Y.flatten()
# Xgrid = np.tile(X.flatten(), nt)
# Ygrid = np.tile(Y.flatten(), nt)
# Tgrid = np.linspace(0, T, nt).repeat(N_plt*N_plt)
# # XX = Variable(torch.from_numpy(Xgrid.reshape(-1,1)).float(), requires_grad = True).to(device)
# # YY = Variable(torch.from_numpy(Ygrid.reshape(-1,1)).float(), requires_grad = True).to(device)
# # TT = Variable(torch.from_numpy(Tgrid.reshape(-1,1)).float(), requires_grad = True).to(device)
# XX = torch.from_numpy(Xgrid.reshape(-1,1)).float().requires_grad_(True)
# YY = torch.from_numpy(Ygrid.reshape(-1,1)).float().requires_grad_(True)
# TT = torch.from_numpy(Tgrid.reshape(-1,1)).float().requires_grad_(True)

#### Adam ####
N_iter_Adam = 51
boundary_num0 = 50
res_num0 = 5000
print("------ IN LBFGS ------")
for i in range(1, len(tlist)):
    print("t = ", tlist[i])
    t_origin = Variable(torch.from_numpy(np.reshape(np.array([0.0] * res_num0), (-1, 1))).float(), requires_grad = True).to(device)
    t_bound = Variable(torch.from_numpy(np.reshape(np.array([tlist[i]] * boundary_num0), (-1, 1))).float(), requires_grad = True).to(device)
    t_res = Variable(torch.from_numpy(np.reshape(np.array([tlist[i]] * res_num0), (-1, 1))).float(), requires_grad = True).to(device)
    for n in range(N_iter_Adam):
        def closure():
            optimizer_LBFGS.zero_grad()
            # optimizer_adam.zero_grad()
            ##### t=0 #####
            x_origin_in = Variable(torch.from_numpy(np.reshape(x_0, (-1, 1))).float(), requires_grad = True).to(device)
            y_origin_in = Variable(torch.from_numpy(np.reshape(y_0, (-1, 1))).float(), requires_grad = True).to(device) 

            u_origin, v_origin, p_origin, f_origin, g_origin, grad_origin = function(net, t_origin, x_origin_in, y_origin_in)
            # zero_1 = torch.zeros(boundary_num0*4 + res_num0, 1).to(device)
            zero_1 = torch.zeros(res_num0, 1).to(device)
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
            zeros_grad = torch.zeros_like(grad_res).to(device)
            f_loss_res = mse_loss_func(f_res, zeros1)       # to satisfy the PDE
            g_loss_res = mse_loss_func(g_res, zeros1)       # to satisfy the PDE
            grad_loss_origin = mse_loss_func(grad_res, zeros_grad)
            loss_res = f_loss_res + g_loss_res + grad_loss_origin

            # LOSS FUNCTION:
            loss = 2 * (loss_origin + loss_up + loss_low + loss_left + loss_right) + loss_res 
            loss.backward()

            return loss
        
        # optimizer_adam.step(closure)
        optimizer_LBFGS.step(closure)
        loss = closure() 
        loss_Adam.append(loss.cpu().detach().numpy())
        
        Tgrid = np.array([tlist[i]] * (N_plt*N_plt))
        XX = Variable(torch.from_numpy(Xgrid.reshape(-1,1)).float(), requires_grad = True).to(device)
        YY = Variable(torch.from_numpy(Ygrid.reshape(-1,1)).float(), requires_grad = True).to(device)
        TT = Variable(torch.from_numpy(Tgrid.reshape(-1,1)).float(), requires_grad = True).to(device)
        
        uu, vv, pp, ff, gg, grad = function(net, TT, XX, YY)
        u = uu.cpu().detach().numpy().reshape(N_plt,N_plt)
        v = vv.cpu().detach().numpy().reshape(N_plt,N_plt)
        p = pp.cpu().detach().numpy().reshape(N_plt,N_plt)
        
        torch.save(net.state_dict(), pth_path)
        N_resolution = 40
        idx = -1
        fig, axs = plt.subplots(2,2, figsize=(7.5, 6), sharey=False)

        N_resolution = 40
        ### Plot u ###
        cp_u = axs[0, 0].contourf(X, Y, u, N_resolution)
        fig.colorbar(cp_u) 
        cp_u.set_cmap('jet')
        axs[0, 0].set_title('Contours of $u$ at t=%f' % tlist[i])
        axs[0, 0].set_xlabel('$x$')
        axs[0, 0].set_ylabel('$y$')

        ### Plot v ###
        cp_v = axs[0, 1].contourf(X, Y, v, N_resolution)
        fig.colorbar(cp_v) 
        cp_v.set_cmap('jet')
        axs[0, 1].set_title('Contours of $v$')
        axs[0, 1].set_xlabel('$x$')
        axs[0, 1].set_ylabel('$y$')

        ### Plot velocity field ###
        strm = axs[1, 0].streamplot(X, Y, u, v, color=v, density=1.5, linewidth=1)
        fig.colorbar(strm.lines)
        strm.lines.set_cmap('jet')
        axs[1, 0].set_title('Velocity stream traces' )
        axs[1, 0].set_xlabel('$x$')
        axs[1, 0].set_ylabel('$y$')

        ### Plot p ###
        cp_p = axs[1, 1].contourf(X, Y, p, N_resolution)
        fig.colorbar(cp_p) 
        cp_p.set_cmap('jet')
        axs[1, 1].set_title('Contours of $p$')
        axs[1, 1].set_xlabel('$x$')
        axs[1, 1].set_ylabel('$y$')

        fig.tight_layout(pad=1.0)
        plt.savefig('figure.png')
        plt.clf()
        plt.close()
        # if n%100 == 0:
        print("Adam - Epoch: ", n, "Training Loss: ", loss.item())
        
        if i==0 and n == 5:
            break
