import torch
from pyDOE import lhs
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import scipy.io
import math, logging, os, glob
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

class Net(nn.Module):

    def __init__(self):
        self.n_hidden = n_hidden
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, self.n_hidden), nn.Tanh(),

            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.Tanh(),

            nn.Linear(self.n_hidden, 2))                    
        
    def forward(self, x):
        return self.net(x)

def function(self, t, x, y):
    nu = 1 / Re   # Re = 100

    # res = self.net(torch.hstack((x, y)))
    res = self(torch.hstack((t, x, y)))
    psi, p = res[:, 0:1], res[:, 1:2]    # NN output: velocity field and pressure field 

    u = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi), create_graph=True)[0].to(device)
    v = -1.*torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0].to(device)

    # u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0].to(device)
    # v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(u), create_graph=True)[0].to(device)

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

    f =  u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    g =  u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)

    return u, v, p, f, g

# def grad(u, v, x, y):
#     u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0].to(device)
#     u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0].to(device)
#     v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0].to(device)
#     v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0].to(device)
#     return u_x, u_y, v_x, v_y

def closure(t0data=0):
    ##### Boundary #####
    boundary_num = x_up_in.shape[0]
    zeros = torch.zeros(boundary_num*nt, 1).to(device)
    ones = torch.ones(boundary_num*nt, 1).to(device)

    tb_in = tli.repeat(boundary_num)
    tbt_in = Variable(torch.from_numpy(np.reshape(tb_in, (-1, 1))).float(), requires_grad = True).to(device)

    # upper boundary 
    u_up, v_up, p_up, f_up, g_up = function(net, tbt_in, x_upt_in, y_upt_in)
    u_loss_up = mask(weight_upt) * mse_loss_func(u_up, ones)           # u = 1 at upper boundary 
    v_loss_up = mask(weight_upt)* mse_loss_func(v_up, zeros)          # v = 0 at upper boundary 
    loss_up = u_loss_up.mean() + v_loss_up.mean()

    # lower boundary 
    u_low, v_low, p_low, f_low, g_low = function(net, tbt_in, x_lowt_in, y_lowt_in)
    u_loss_low = mask(weight_lowt) * mse_loss_func(u_low, zeros)        # u = 0 at lower boundary 
    v_loss_low = mask(weight_lowt) * mse_loss_func(v_low, zeros)        # v = 0 at lower boundary 
    loss_low = u_loss_low.mean() + v_loss_low.mean()

    # left boundary 
    u_left, v_left, p_left, f_left, g_left = function(net, tbt_in, x_leftt_in, y_leftt_in)
    u_loss_left = mask(weight_leftt) *mse_loss_func(u_left, zeros)      # u = 0 at lower boundary
    v_loss_left = mask(weight_leftt) * mse_loss_func(v_left, zeros)      # v = 0 at lower boundary
    loss_left = u_loss_left.mean() + v_loss_left.mean()

    # right boundary 
    u_right, v_right, p_right, f_right, g_right = function(net, tbt_in, x_rightt_in, y_rightt_in)
    u_loss_right = mask(weight_rightt) * mse_loss_func(u_right, zeros)        # u = 0 at lower boundary
    v_loss_right = mask(weight_rightt) * mse_loss_func(v_right, zeros)        # v = 0 at lower boundary
    loss_right = u_loss_right.mean() + v_loss_right.mean()

    ##### Residual #####
    # res_num = x_res_in.shape[0] / nt
    tbres_in = tli.repeat(res_num)
    tbrest_in = Variable(torch.from_numpy(np.reshape(tbres_in, (-1, 1))).float(), requires_grad = True).to(device)

    u_res, v_res, p_res, f_res, g_res = function(net, tbrest_in, x_rest_in, y_rest_in)
    zeros1 = t0data
    
    ut_res = u_res.reshape(nt, res_num)
    vt_res = v_res.reshape(nt, res_num)
    ft_res = f_res.reshape(nt, res_num)
    gt_res = g_res.reshape(nt, res_num)
    # for i in range(nt):
    #     A = torch.from_numpy(amat[i].repeat(repeats=res_num)).to(device)
    uci = ut_res + dt * A @ ft_res
    vci = vt_res + dt * A @ gt_res
    
    u_loss_res = mask(weight_rest) * mse_loss_func(uci.reshape(-1, 1), zeros1[0].reshape(-1,1)) 
    v_loss_res = mask(weight_rest) * mse_loss_func(vci.reshape(-1, 1), zeros1[1].reshape(-1,1))
    loss_res = u_loss_res.mean() + v_loss_res.mean()


    # f_loss_res = mse_loss_func(f_res, zeros1)       # to satisfy the PDE
    # g_loss_res = mse_loss_func(g_res, zeros1)       # to satisfy the PDE
    # loss_up_val = loss_up.item()
    # loss_low_val = loss_low.item()
    # loss_left_val = loss_left.item()
    # loss_right_val = loss_right.item()
    # loss_res_val = loss_res.item()

    # LOSS FUNCTION:
    # torch.autograd.set_detect_anomaly(True)
    loss =  loss_up + loss_low + loss_left + loss_right + loss_res
    
    return loss

def find_uvp(t, X, Y, N_plt):
    TT = Variable(torch.from_numpy(t.reshape(-1,1)).float(), requires_grad = True).to(device)
    XX = Variable(torch.from_numpy(X.reshape(-1,1)).float(), requires_grad = True).to(device)
    YY = Variable(torch.from_numpy(Y.reshape(-1,1)).float(), requires_grad = True).to(device)
    uu, vv, pp, ff, gg = function(net, TT, XX, YY)
    u = uu.cpu().detach().numpy().reshape(-N_plt,N_plt)
    v = vv.cpu().detach().numpy().reshape(-N_plt,N_plt)
    p = pp.cpu().detach().numpy().reshape(-N_plt,N_plt)
    return u, v, p

def mask(weight, c=1, k=1.5):
    return c * weight ** k

class MyOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MyOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                p.data.add_(group['lr'], d_p)  # change here

class MyAdam(torch.optim.Adam):
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # step_size = group['lr'] * (bias_correction2.sqrt() / bias_correction1)
                step_size = torch.div(torch.mul(group['lr'], math.sqrt(bias_correction2)), bias_correction1)

                p.data.add_(step_size, exp_avg.div(denom))  # change here

if __name__ == '__main__':
    np.random.seed(1234)
    Re = 500
    n_hidden = 128
    out_dir = "outputs"

    os.makedirs(out_dir, exist_ok=True)
    existing_dirs = glob.glob(os.path.join(out_dir, 'out*'))
    num_existing_dirs = len(existing_dirs)
    out_dir = os.path.join(out_dir, f'out{num_existing_dirs + 1}')
    os.makedirs(out_dir, exist_ok=True)
    print(f'Created new directory: {out_dir}')
    log_path = os.path.join(out_dir, 'myresult.log')

    print("Cuda availability: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    
    ## logger 
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ##### Boundary points #####
    boundary_num = 1000

    # upper boundary
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

    x_res = res_pts[:, 0]
    y_res = res_pts[:, 1]
    x_res_in = Variable(torch.from_numpy(np.reshape(x_res, (-1, 1))).float(), requires_grad = True).to(device)
    y_res_in = Variable(torch.from_numpy(np.reshape(y_res, (-1, 1))).float(), requires_grad = True).to(device)
    
    ### time
    dt = 0.1
    Tmax = 50
    
    q = 5
    tmp = np.float32(np.loadtxt('Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
    IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
    amat = IRK_weights[0:-1]
    bvec = IRK_weights[-1]
    cvec = amat.sum(1)

    A = torch.from_numpy(amat).to(device)
    B = torch.from_numpy(bvec).reshape(1,-1).to(device)


    ##### Training #####
    net = Net()
    print(net)
    net = net.to(device)

    ##### Xavier Initialization #####
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            torch.nn.init.xavier_normal_(m.weight)


    ##### Loss Function #####
    mse_loss_func = torch.nn.MSELoss(reduction='none')
    optimizer_adam = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_Adam = []

    optimizer_LBFGS = torch.optim.LBFGS(net.parameters(), history_size=8, max_iter=500000)
    loss_LBFGS = []

    start_t = 0
    end_t = dt
    nt = q
    
    x_up_t = np.tile(x_up, (nt, 1))
    y_up_t = np.tile(y_up, (nt, 1))
    x_low_t = np.tile(x_low, (nt, 1))
    y_low_t = np.tile(y_low, (nt, 1))
    x_left_t = np.tile(x_left, (nt, 1))
    y_left_t = np.tile(y_left, (nt, 1))
    x_right_t = np.tile(x_right, (nt, 1))
    y_right_t = np.tile(y_right, (nt, 1))
    x_res = np.tile(res_pts[:, 0], (nt, 1))
    y_res = np.tile(res_pts[:, 1], (nt, 1))
    x_upt_in = Variable(torch.from_numpy(np.reshape(x_up_t, (-1, 1))).float(), requires_grad = True).to(device)
    y_upt_in = Variable(torch.from_numpy(np.reshape(y_up_t, (-1, 1))).float(), requires_grad = True).to(device)
    x_rest_in = Variable(torch.from_numpy(np.reshape(x_res, (-1, 1))).float(), requires_grad = True).to(device)
    y_rest_in = Variable(torch.from_numpy(np.reshape(y_res, (-1, 1))).float(), requires_grad = True).to(device)
    x_lowt_in = Variable(torch.from_numpy(np.reshape(x_low_t, (-1, 1))).float(), requires_grad = True).to(device)
    y_lowt_in = Variable(torch.from_numpy(np.reshape(y_low_t, (-1, 1))).float(), requires_grad = True).to(device)
    x_leftt_in = Variable(torch.from_numpy(np.reshape(x_left_t, (-1, 1))).float(), requires_grad = True).to(device)
    y_leftt_in = Variable(torch.from_numpy(np.reshape(y_left_t, (-1, 1))).float(), requires_grad = True).to(device)
    x_rightt_in = Variable(torch.from_numpy(np.reshape(x_right_t, (-1, 1))).float(), requires_grad = True).to(device)
    y_rightt_in = Variable(torch.from_numpy(np.reshape(y_right_t, (-1, 1))).float(), requires_grad = True).to(device)

    
    weight_upt = Variable(torch.ones_like(x_upt_in) * 3, requires_grad = True).to(device)
    weight_lowt = Variable(torch.ones_like(x_lowt_in), requires_grad = True).to(device)
    weight_leftt = Variable(torch.ones_like(x_leftt_in), requires_grad = True).to(device)
    weight_rightt = Variable(torch.ones_like(x_rightt_in), requires_grad = True).to(device)
    # weight_rest = Variable(torch.ones_like(x_rest_in), requires_grad = True).to(device)
    # weight_lowt = torch.ones_like(x_lowt_in).to(device)
    # weight_leftt = torch.ones_like(x_leftt_in).to(device)
    # weight_rightt = torch.ones_like(x_rightt_in).to(device)
    weight_rest = torch.ones_like(x_rest_in).to(device)

    # torch.nn.init.uniform_(weight_rest)
    # torch.nn.init.uniform_(weight_rightt)
    # torch.nn.init.uniform_(weight_leftt)
    # torch.nn.init.uniform_(weight_lowt)
    # torch.nn.init.uniform_(weight_upt)

    t0data = torch.zeros(2, res_num*nt).to(device)

    ######### Training ########
    
    for i in range(499):
        start_t = (i+1) * dt
        end_t = (i+2) * dt
        # tli = np.arange(start_t, end_t, nt) * cvec
        tli = cvec.copy() * dt + start_t
        
        stli = np.array([start_t] * res_num)
        t0 = Variable(torch.from_numpy(stli.reshape(-1, 1)).float(), requires_grad = True).to(device)
        xres0 = Variable(torch.from_numpy(np.reshape(res_pts[:, 0], (-1, 1))).float(), requires_grad = True).to(device)
        yres0 = Variable(torch.from_numpy(np.reshape(res_pts[:, 1], (-1, 1))).float(), requires_grad = True).to(device)

        # tbres_in = tli.repeat(res_num)
        # tbrest_in = Variable(torch.from_numpy(np.reshape(tbres_in, (-1, 1))).float(), requires_grad = True).to(device)
        
        #### Adam ####
        N_iter_Adam = 10001
        print("------ IN Adam ------")
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        optimizer_lamda = torch.optim.Adam([
        # optimizer_lamda = MyAdam([
            # weight_upt], lr=5e-2)
            weight_upt, weight_lowt, weight_leftt, weight_rightt], lr=5e-3)
        for n in range(N_iter_Adam):
            optimizer_lamda.zero_grad()
            loss2 = torch.mul(closure(t0data), -1).to(device)
            loss2.backward()
            # loss2 = -loss
            optimizer_lamda.step()

            optimizer.zero_grad()
            loss = closure(t0data) 
            loss.backward()
            optimizer.step()

            
            
            loss_Adam.append(loss.cpu().detach().numpy())

            if n%10 == 0:
                logger.info(
                "Adam - Epoch: %d, weight_upt: %.6f±%.6f, weight_lowt: %.6f±%.6f, weight_rest: %.6f±%.6f, lr of MyAdam: %.6f, Training Loss: %.6f",
                    n, mask(weight_upt).mean().item(), mask(weight_upt).std().item(), mask(weight_lowt).mean().item(), mask(weight_lowt).std().item(), mask(weight_rest).mean().item(), mask(weight_rest).std().item(), optimizer_lamda.param_groups[0]['lr'], loss.item())

            if n%100 == 0:
                print(
                    "Adam - Epoch: ", n, 
                    "weight_upt: {:.6f}±{:.6f}".format(mask(weight_upt).mean().item(), mask(weight_upt).std().item()),
                    "weight_lowt: {:.6f}±{:.6f}".format(mask(weight_lowt).mean().item(), mask(weight_lowt).std().item()),
                    "weight_rest: {:.6f}±{:.6f}".format(mask(weight_rest).mean().item(), mask(weight_rest).std().item()),
                    "lr of MyAdam: {:.6f}".format(optimizer_lamda.param_groups[0]['lr']),
                    "Training Loss: ", loss.item())
                
                
                fig, axs = plt.subplots(2, 2, figsize=(7.5, 6), sharey=False)
                N_plt = 200

                xlist = np.linspace(0., 1., N_plt)
                ylist = np.linspace(0., 1., N_plt)
                X, Y = np.meshgrid(xlist, ylist)
                T = np.array([start_t] * X.size)
                u, v, p = find_uvp(T, X, Y, N_plt)

                N_resolution = 40
                ### Plot u ###
                cp_u = axs[0, 0].contourf(X, Y, u, N_resolution)
                fig.colorbar(cp_u) 
                cp_u.set_cmap('jet')
                axs[0, 0].set_title('Contours of $u$')
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
                plt.savefig(os.path.join(out_dir , "Results_%.1f.png"%start_t))
                plt.close()
            

        #### LBFGS ####
        print("------ IN LBFGS ------")
        N_iter_LBFGS = 200
        optimizer = torch.optim.LBFGS(net.parameters(), history_size=8, max_iter=500000)
        # optimizer_lamda = torch.optim.LBFGS([
        # optimizer_lamda = MyAdam([
            # weight_upt, weight_lowt, weight_leftt, weight_rightt, weight_rest], history_size=8, max_iter=500000)
        for n in range(N_iter_LBFGS):
            loss = closure(t0data) 
            optimizer.step()
            # optimizer_lamda.step()
            loss_LBFGS.append(loss.cpu().detach().numpy())
            
            print("LBFGS - Epoch: ", n, "Training Loss: ", loss.item())
            fig, axs = plt.subplots(2, 2, figsize=(7.5, 6), sharey=False)
            N_plt = 200

            xlist = np.linspace(0., 1., N_plt)
            ylist = np.linspace(0., 1., N_plt)
            X, Y = np.meshgrid(xlist, ylist)
            T = np.array([start_t] * X.size)
            u, v, p = find_uvp(T, X, Y, N_plt)

            N_resolution = 40
            ### Plot u ###
            cp_u = axs[0, 0].contourf(X, Y, u, N_resolution)
            fig.colorbar(cp_u) 
            cp_u.set_cmap('jet')
            axs[0, 0].set_title('Contours of $u$')
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
            plt.savefig("Results.png")
            plt.close()

        # predict for next time step
        
        # u0, v0, p0, f0, g0 = function(net, t0, xres0, yres0)
        # u0s = torch.tile(u0.T, (nt, 1)).reshape(-1, 1).T
        # v0s = torch.tile(v0.T, (nt, 1)).reshape(-1, 1).T
        # t0data = torch.vstack((u0s, v0s))

        tbres_in = tli.repeat(res_num)
        tbrest_in = Variable(torch.from_numpy(np.reshape(tbres_in, (-1, 1))).float(), requires_grad = True).to(device)

        u_res, v_res, p_res, f_res, g_res = function(net, tbrest_in, x_rest_in, y_rest_in)            
        ut_res = u_res.reshape(nt, res_num)
        vt_res = v_res.reshape(nt, res_num)
        ft_res = f_res.reshape(nt, res_num)
        gt_res = g_res.reshape(nt, res_num)
        # for i in range(nt):
        #     A = torch.from_numpy(amat[i].repeat(repeats=res_num)).to(device)
        t0data -= dt * B @ ft_res
        t0data -= dt * B @ gt_res

        ##### Save the model #####
        torch.save(net.state_dict(), 'model_RK_Re%d.pth'%Re)

        fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=False)

        ep1 = np.arange(50, N_iter_Adam)
        ep2 = np.arange(N_iter_Adam - 1, N_iter_Adam + N_iter_LBFGS)

        loss_LBFGS1 = [loss_Adam[-1]] + loss_LBFGS
        axs[0].semilogy(ep1, loss_Adam[50:], label="Adam Loss")
        axs[0].semilogy(ep2, loss_LBFGS1, label="LBFGS Loss")
        axs[0].legend()
        axs[0].set_title('Training Loss' )
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')

        ep3 = np.arange(9900, N_iter_Adam)
        ep4 = np.arange(N_iter_Adam - 1, N_iter_Adam + N_iter_LBFGS)
        axs[1].semilogy(ep3, loss_Adam[9900:], label="Adam Loss")
        axs[1].semilogy(ep4, loss_LBFGS1, label="LBFGS Loss")
        axs[1].legend()
        axs[1].set_title('Training Loss (last 100 iterations)' )
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')

        fig.tight_layout(pad=1.0)
        plt.savefig("Loss_%d.png"%i)
        plt.close()

        
