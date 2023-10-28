import os
import torch
import torch.nn as nn
import torch.optim as optim

from time import time
from datetime import datetime

from agent.actor import PtrNet1
from agent.critic import PtrNet2
from environment.env import PanelBlockShop


# torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True


def train_model(env, params, log_path=None):
    date = datetime.now().strftime('%m%d_%H_%M')
    param_path = params["log_dir"] + '/%s_%s_param.csv' % (date, "train")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

    act_model = PtrNet1(params)
    if params["load_model"]:
        path = params["model_dir"] + "/sl/" + max(os.listdir(params["model_dir"] + "/sl"))
        act_model.load_state_dict(torch.load(path))
        act_model.train()

    if params["optimizer"] == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])
    if params["is_lr_decay"]:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    act_model = act_model.to(device)
    ave_act_loss = 0.0

    if params["use_critic"]:
        cri_model = PtrNet2(params)
        if params["optimizer"] == 'Adam':
            cri_optim = optim.Adam(cri_model.parameters(), lr=params["lr"])
        elif params["optimizer"] == 'RMSProp':
            cri_optim = optim.RMSprop(cri_model.parameters(), lr=params["lr"])
        if params["is_lr_decay"]:
            cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, step_size=params["lr_decay_step"],
                                                         gamma=params["lr_decay"])
        cri_model = cri_model.to(device)
        mse_loss = nn.MSELoss()
        ave_cri_loss = 0.0

    ave_makespan = 0.0

    t1 = time()
    for s in range(params["step"]):
        inputs_temp = env.generate_data(params["batch_size"])
        inputs = inputs_temp / inputs_temp.amax(dim=(1,2)).unsqueeze(-1).unsqueeze(-1)\
            .expand(-1, inputs_temp.shape[1], inputs_temp.shape[2])

        pred_seq, ll, _ = act_model(inputs, device)
        real_makespan = env.stack_makespan(inputs_temp, pred_seq)

        if params["use_critic"]:
            pred_makespan = cri_model(inputs, device)
            cri_loss = mse_loss(pred_makespan, real_makespan.detach())
            cri_optim.zero_grad()
            cri_loss.backward()
            nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm=1., norm_type=2)
            cri_optim.step()
            if params["is_lr_decay"]:
                cri_lr_scheduler.step()
            ave_cri_loss += cri_loss.item()
        else:
            if s == 0:
                makespan = real_makespan.detach().mean()
            else:
                makespan = (makespan * 0.9) + (0.1 * real_makespan.detach().mean())
            pred_makespan = makespan

        adv = real_makespan.detach() - pred_makespan.detach()
        act_loss = (adv * ll).mean()
        act_optim.zero_grad()
        act_loss.backward()
        act_optim.step()
        nn.utils.clip_grad_norm_(act_model.parameters(), max_norm=1., norm_type=2)
        if params["is_lr_decay"]:
            act_lr_scheduler.step()
        ave_act_loss += act_loss.item()

        ave_makespan += real_makespan.mean().item()

        if s % params["log_step"] == 0:
            t2 = time()
            if params["use_critic"]:
                print('step:%d/%d, actic loss:%1.3f, critic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                    s, params["step"], ave_act_loss / (s + 1), ave_cri_loss / (s + 1), ave_makespan / (s + 1),
                    (t2 - t1) // 60, (t2 - t1) % 60))
                if log_path is None:
                    log_path = params["log_dir"] + '/%s_train.csv' % date
                    with open(log_path, 'w') as f:
                        f.write('step,actic loss,critic loss,average makespan,time\n')
                else:
                    with open(log_path, 'a') as f:
                        f.write('%d,%1.4f,%1.4f,%dmin%dsec\n' % (s, ave_act_loss / (s + 1), ave_makespan / (s + 1),
                                                                 (t2 - t1) // 60, (t2 - t1) % 60))
            else:
                print('step:%d/%d, actic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                    s, params["step"], ave_act_loss / (s + 1), ave_makespan / (s + 1), (t2 - t1) // 60, (t2 - t1) % 60))
                if log_path is None:
                    log_path = params["log_dir"] + '/%s_train.csv' % date
                    with open(log_path, 'w') as f:
                        f.write('step,actic loss,average makespan,time\n')
                else:
                    with open(log_path, 'a') as f:
                        f.write('%d,%1.4f,%1.4f,%dmin%dsec\n' % (s, ave_act_loss / (s + 1), ave_makespan / (s + 1),
                                                                 (t2 - t1) // 60, (t2 - t1) % 60))
            t1 = time()

        if s % params["save_step"] == 0:
            torch.save(act_model.state_dict(), params["model_dir"] + '/rl' + '/%s_step%d_act.pt' % (date, s))
            print('save model...')


if __name__ == '__main__':

    load_model = False

    log_dir = "./result/log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model_dir = "./result/model"
    if not os.path.exists(model_dir + "/rl"):
        os.makedirs(model_dir + "/rl")

    params = {
        "num_of_process": 6,
        "num_of_blocks": 40,
        "step": 50001,
        "log_step": 10,
        "log_dir": log_dir,
        "save_step": 1000,
        "model_dir": model_dir,
        "batch_size": 64,
        "n_embedding": 512,
        "n_hidden": 512,
        "init_min": -0.08,
        "init_max": 0.08,
        "clip_logits": 20,
        "softmax_T": 1.0,
        "decode_type": "sampling",
        "optimizer": "Adam",
        "n_glimpse": 1,
        "n_process": 3,
        "lr": 1e-5,
        "is_lr_decay": True,
        "lr_decay": 0.98,
        "lr_decay_step": 2000,
        "use_critic": False,
        "load_model": load_model
    }

    env = PanelBlockShop(params["num_of_process"], params["num_of_blocks"])
    train_model(env, params)