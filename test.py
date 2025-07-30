from main_PPO import *
import os

if __name__ == '__main__':

    load_model = False

    log_dir = "./result/log"
    if not os.path.exists(log_dir + "/ppo"):
        os.makedirs(log_dir + "/ppo")

    model_dir = "./result/model"
    if not os.path.exists(model_dir + "/ppo_w_third_feature"):
        os.makedirs(model_dir + "/ppo_w_third_feature")

    load_model = False

    log_dir = "./result/log"
    if not os.path.exists(log_dir + "/ppo"):
        os.makedirs(log_dir + "/ppo")

    model_dir = "./result/model"
    if not os.path.exists(model_dir + "/ppo_w_third_feature"):
        os.makedirs(model_dir + "/ppo_w_third_feature")

    param0 ={
             "alpha": 0.1,
             "n_hidden": 108,
             "ex_embedding_size": 36,
             "ex_embedding_size2": 48,
             "n_multi_head": 2,
             "k_hop": 1,
             "lr_latent": 5.0e-5,
             "lr_critic": 1.0e-4,
             "lr": 1.0e-4,
             "entropy_coeff": 0.0001,
             "layers": eval('[256, 128]'),
             "lr_decay_step": 500,
             "lr_decay": 0.95,
             "lr_decay_min": 5e-5
             }

    param1 ={
             "alpha": 0.05,
             "n_hidden": 128,
             "ex_embedding_size": 32,
             "ex_embedding_size2": 54,
             "n_multi_head": 1,
             "k_hop": 1,
             "lr_latent": 5.0e-5,
             "lr_critic": 1.0e-4,
             "lr": 1.0e-4,
             "entropy_coeff": 0.001,
             "layers": eval('[196, 96]'),
             "lr_decay_step": 400,
             "lr_decay": 0.99,
             "lr_decay_min": 5e-5
             }

    param2 ={
             "alpha": 0.05,
             "n_hidden": 128,
             "ex_embedding_size": 32,
             "ex_embedding_size2": 54,
             "n_multi_head": 1,
             "k_hop": 1,
             "lr_latent": 5.0e-5,
             "lr_critic": 1.0e-4,
             "lr": 1.0e-4,
             "entropy_coeff": 0.00001,
             "layers": eval('[196, 96]'),
             "lr_decay_step": 1000,
             "lr_decay": 0.99,
             "lr_decay_min": 5e-5
             }

    param3 ={
             "alpha": 0.1,
             "n_hidden": 144,
             "ex_embedding_size": 36,
             "ex_embedding_size2": 56,
             "n_multi_head": 2,
             "k_hop": 1,
             "lr_latent": 5.0e-5,
             "lr_critic": 1.0e-4,
             "lr": 1.0e-4,
             "entropy_coeff": 0.0005,
             "layers": eval('[512, 256, 128]'),
             "lr_decay_step": 700,
             "lr_decay": 0.995,
             "lr_decay_min": 2.5e-5
             } #나가리

    param4 ={
             "alpha": 0.1,
             "n_hidden": 256,
             "ex_embedding_size": 42,
             "ex_embedding_size2": 56,
             "n_multi_head": 3,
             "k_hop": 1,
             "lr_latent": 5.0e-5,
             "lr_critic": 1.0e-4,
             "lr": 1.0e-4,
             "entropy_coeff": 0.00005,
             "layers": eval('[128, 64]'),
             "lr_decay_step": 500,
             "lr_decay": 0.975,
             "lr_decay_min": 2.5e-5
             }


    param5 ={
             "alpha": 0.1,
             "n_hidden": 196,
             "ex_embedding_size": 36,
             "ex_embedding_size2": 64,
             "n_multi_head": 3,
             "k_hop": 1,
             "lr_latent": 5.0e-5,
             "lr_critic": 1.0e-4,
             "lr": 1.0e-4,
             "entropy_coeff": 0.0005,
             "layers": eval('[196, 128]'),
             "lr_decay_step": 1000,
             "lr_decay": 0.95,
             "lr_decay_min": 2.5e-5
             }

    selected_param = str(os.environ.get("selected_param", "param4"))
    param_group = {"param0": param0,
                   "param1": param1,
                   "param2": param2,
                   "param3": param3,
                   "param4": param4,
                   "param5": param5,
                   }
    print("입력된 파라미터 :", selected_param)
    param = param_group[selected_param]

    params = {
        "num_of_process": 6,
        "step": cfg.step,
        "log_step": cfg.log_step,
        "log_dir": log_dir,
        "save_step": cfg.save_step,
        "model_dir": model_dir,
        "batch_size": cfg.batch_size,
        "init_min": -0.08,
        "init_max": 0.08,
        "use_logit_clipping": True,
        "C": cfg.C,
        "T": cfg.T,
        "iteration": cfg.iteration,
        "epsilon": float(os.environ.get("epsilon", 0.2)),
        "optimizer": "Adam",
        "n_glimpse": cfg.n_glimpse,
        "n_process": cfg.n_process,
        "lr_decay_step_critic": cfg.lr_decay_step_critic,
        "load_model": load_model,
        "dot_product": cfg.dot_product,
        "reward_scaler": cfg.reward_scaler,
        "beta": float(os.environ.get("beta", 0.65)),
        "alpha": param["alpha"],
        "lr_latent": param['lr_latent'],
        "lr_critic": param['lr_critic'],
        "lr":  param['lr'],
        "lr_decay":param["lr_decay"],
        "lr_decay_step": param['lr_decay_step'] ,
        "lr_decay_min": param['lr_decay_min'],
        "layers": param['layers'],
        "n_embedding": 48,
        "n_hidden": param["n_hidden"],
        "graph_embedding_size": int(os.environ.get("graph_embedding_size", 96)),
        "n_multi_head": param["n_multi_head"],
        "ex_embedding_size": param["ex_embedding_size"],
        "ex_embedding_size2": param["ex_embedding_size2"],
        "entropy_coeff": param["entropy_coeff"],

        "k_hop": int(os.environ.get("k_hop", 1)),
        "is_lr_decay": True,
        "third_feature": 'first_and_second',  # first_and_second, first_only, second_only
        "baseline_reset": True,
        "ex_embedding": True,
        "w_representation_learning": True,
        "z_dim": 128,
        "k_epoch": int(os.environ.get("k_epoch", 1)),
        "target_entropy": int(os.environ.get("target_entropy", -2)),

    }

    test_model(params, selected_param)  #f


    flow_shop_index_list = list()
    bottleneck_index_list = list()
    h_list = list()
    z_list = list()

    for i in range(5000):
        num_machine = np.random.randint(5, 10)
        num_job = np.random.randint(num_machine, 10)
        jobs_datas, scheduler_list = generate_jssp_instance(num_jobs=num_job, num_machine=num_machine, batch_size=1)
        act_model.Latent.current_num_edges = num_machine * num_job
        flow_shop_index = calculate_flow_shop_index(scheduler_list[0])
        bottleneck_index = calculate_bottleneck_index(scheduler_list[0])
        act_model.get_jssp_instance(scheduler_list)
        scheduler = AdaptiveScheduler(jobs_datas[0])
        node_features = [scheduler.get_node_feature()]
        edge_precedence = scheduler.get_edge_index_precedence()
        edge_antiprecedence = scheduler.get_edge_index_antiprecedence()
        edge_machine_sharing = scheduler.get_machine_sharing_edge_index()
        heterogeneous_edges = [(edge_precedence, edge_antiprecedence, edge_machine_sharing)]  # 세종류의 엣지들을 하나의 변수로 참조시킴
        input_data = (node_features, heterogeneous_edges)
        z, v, h, pred_seq = act_model.forward_visualize(input_data,
                                     device,
                                     scheduler_list=scheduler_list,
                                     num_machine=num_machine,
                                     num_job=num_job
                                     )


        sequence = pred_seq[0]
        scheduler = AdaptiveScheduler(jobs_datas[0])
        makespan = -scheduler.run(sequence.tolist())

        #print(z.shape)
        flow_shop_index_list.append(v.squeeze(0).detach().cpu().numpy()[0])
        bottleneck_index_list.append(makespan)
        h_list.append(h.squeeze(0).detach().cpu().numpy().tolist())
        z_list.append(z.squeeze(0).detach().cpu().numpy().tolist())
        print("data collecting : ", i)

    # numpy 배열로 변환
    flow_shop_index_list = np.array(flow_shop_index_list)
    bottleneck_index_list = np.array(bottleneck_index_list)
    h_array = np.array(h_list)  # z_list를 빈 리스트로 재정의하지 않고 numpy 배열로 변환
    z_array = np.array(z_list)  # z_list를 빈 리스트로 재정의하지 않고 numpy 배열로 변환

    np.save('flow_shop_index_list.npy', flow_shop_index_list)
    np.save('bottleneck_index_list.npy', bottleneck_index_list)
    np.save('h_array.npy', h_array)
    np.save('z_array.npy', z_array)
