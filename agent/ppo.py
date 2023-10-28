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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    date = datetime.now().strftime('%m%d_%H_%M')
    #print(params)
    param_path = params["log_dir"] + '/ppo' + '/%s_%s_param.csv' % (date, "train")
    print(f'generate {param_path}')
    with open(param_path, 'w') as f:
        f.write(''.join('%s,%s\n' % item for item in params.items()))

    epoch = 0
    ave_act_loss = 0.0
    ave_cri_loss = 0.0
    ave_makespan = 0.0
    act_model = PtrNet1(params).to(device)
    cri_model = PtrNet2(params).to(device)
    if params["optimizer"] == 'Adam':
        act_optim = optim.Adam(act_model.parameters(), lr=params["lr"])
        cri_optim = optim.Adam(cri_model.parameters(), lr=params["lr"])
    elif params["optimizer"] == "RMSProp":
        act_optim = optim.RMSprop(act_model.parameters(), lr=params["lr"])
        cri_optim = optim.RMSprop(cri_model.parameters(), lr=params["lr"])

    if params["load_model"]:
        checkpoint = torch.load(params["model_dir"] + "/ppo/" + "0821_21_36_step100000_act.pt")
        act_model.load_state_dict(checkpoint['model_state_dict_actor'])
        cri_model.load_state_dict(checkpoint['model_state_dict_critic'])
        act_optim.load_state_dict(checkpoint['optimizer_state_dict_actor'])
        cri_optim.load_state_dict(checkpoint['optimizer_state_dict_critic'])
        epoch = checkpoint['epoch']
        ave_act_loss = checkpoint['ave_act_loss']
        ave_cri_loss = checkpoint['ave_cri_loss']
        ave_makespan = checkpoint['ave_makespan']
        act_model.train()
        cri_model.train()

    if params["is_lr_decay"]:
        act_lr_scheduler = optim.lr_scheduler.StepLR(act_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])
        cri_lr_scheduler = optim.lr_scheduler.StepLR(cri_optim, step_size=params["lr_decay_step"],
                                                     gamma=params["lr_decay"])

    mse_loss = nn.MSELoss()

    t1 = time()
    for s in range(epoch + 1, params["step"]):
        inputs_temp = env.generate_data(params["batch_size"])
        #print(inputs_temp.shape, inputs_temp.amax(dim=(1,2)).shape)
        if env.distribution == "lognormal":
            inputs = inputs_temp / inputs_temp.amax(dim=(1,2)).unsqueeze(-1).unsqueeze(-1)\
                .expand(-1, inputs_temp.shape[1], inputs_temp.shape[2])
        elif env.distribution == "uniform":
            inputs = inputs_temp / 100
        """
        변수별 shape 
        inputs : batch_size X number_of_blocks X number_of_process
        pred_seq : batch_size X number_of_blocks 
        
        """

        pred_seq, ll_old, _ = act_model(inputs, device)

        # pred_seq :

        for k in range(params["iteration"]):                              # K-epoch
            real_makespan = env.stack_makespan(inputs_temp, pred_seq)
            pred_makespan = cri_model(inputs, device).unsqueeze(-1)       # predicted value(non_discounted)라고 봐야할 것 같음. cri_model은 pred_seq가 입력되는 것이 아니라 그냥 input(아마 state에 해당한다고 볼 수 있을 것 같음)이 들어간다
            adv = real_makespan.detach() - pred_makespan.detach()         # 참고로 일반적인 PPO에서는 TD target(여기서는 real_makespan에 해당) 구할 때 reward + gamma * v(state) 방식으로 구함
            cri_loss = mse_loss(pred_makespan, real_makespan.detach())    # MSE loss를 구한다.
            cri_optim.zero_grad()
            cri_loss.backward()
            nn.utils.clip_grad_norm_(cri_model.parameters(), max_norm=1., norm_type=2)
            cri_optim.step()
            if params["is_lr_decay"]:
                cri_lr_scheduler.step()
            ave_cri_loss += cri_loss.item()
            _, ll_new, _ = act_model(inputs, device, pred_seq)            # pi(seq|inputs)
            ratio = torch.exp(ll_new - ll_old.detach()).unsqueeze(-1)     #
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - params["epsilon"], 1 + params["epsilon"]) * adv
            act_loss = torch.max(surr1, surr2).mean()
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
            print('step:%d/%d, actic loss:%1.3f, crictic loss:%1.3f, L:%1.3f, %dmin%dsec' % (
                s, params["step"], ave_act_loss / ((s + 1) * params["iteration"]), ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan / (s + 1), (t2 - t1) // 60, (t2 - t1) % 60))
            if log_path is None:
                log_path = params["log_dir"] + '/ppo' + '/%s_train.csv' % date
                with open(log_path, 'w') as f:
                    f.write('step,actic loss, crictic loss, average makespan,time\n')
            else:
                with open(log_path, 'a') as f:
                    f.write('%d,%1.4f,%1.4f,%1.4f,%dmin%dsec\n' % (s, ave_act_loss / ((s + 1) * params["iteration"]), ave_cri_loss / ((s + 1) * params["iteration"]), ave_makespan / (s + 1),
                                                             (t2 - t1) // 60, (t2 - t1) % 60))
            t1 = time()

        if s % params["save_step"] == 0:
            torch.save({'epoch': s,
                        'model_state_dict_actor': act_model.state_dict(),
                        'model_state_dict_critic': cri_model.state_dict(),
                        'optimizer_state_dict_actor': act_optim.state_dict(),
                        'optimizer_state_dict_critic': cri_optim.state_dict(),
                        'ave_act_loss': ave_act_loss,
                        'ave_cri_loss': ave_cri_loss,
                        'ave_makespan': ave_makespan},
                       params["model_dir"] + '/ppo' + '/%s_step%d_act.pt' % (date, s))
            print('save model...')


if __name__ == '__main__':

    load_model = False

    log_dir = "./result/log"
    if not os.path.exists(log_dir + "/ppo"):
        os.makedirs(log_dir + "/ppo")

    model_dir = "./result/model"
    if not os.path.exists(model_dir + "/ppo"):
        os.makedirs(model_dir + "/ppo")

    params = {
        "num_of_process": 6,
        "num_of_blocks": 40,
        "step": 150001,
        "log_step": 10,
        "log_dir": log_dir,
        "save_step": 1000,
        "model_dir": model_dir,
        "batch_size": 128,
        "n_embedding": 1024,
        "n_hidden": 512,
        "init_min": -0.08,
        "init_max": 0.08,
        "use_logit_clipping": True,
        "C": 10,
        "T": 1.0,
        "decode_type": "sampling",
        "iteration": 2,
        "epsilon": 0.2,
        "optimizer": "Adam",
        "n_glimpse": 1,
        "n_process": 3,
        "lr": 1e-4,
        "is_lr_decay": True,
        "lr_decay": 0.98,
        "lr_decay_step": 2000,
        "load_model": load_model
    }

    env = PanelBlockShop(params["num_of_process"], params["num_of_blocks"], distribution="lognormal")
    train_model(env, params)