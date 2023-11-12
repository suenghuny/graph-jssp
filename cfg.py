import argparse

def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vessl", type=bool, default=False, help="vessl AI 사용여부")
    parser.add_argument("--step", type=int, default=400001, help="")
    parser.add_argument("--log_step", type=int, default=10, help="")

    parser.add_argument("--save_step", type=int, default=40, help="")
    parser.add_argument("--batch_size", type=int, default=32, help="")
    parser.add_argument("--C", type=float, default=10, help="")
    parser.add_argument("--T", type=float, default=1.0, help="")
    parser.add_argument("--iteration", type=int, default=2, help="")
    parser.add_argument("--epsilon", type=float, default=0.2, help="")
    parser.add_argument("--n_glimpse", type=int, default=4, help="")
    parser.add_argument("--n_process", type=int, default=4, help="")
    parser.add_argument("--lr", type=float, default=1.e-3, help="")
    parser.add_argument("--lr_critic", type=float, default=2.e-4, help="")
    parser.add_argument("--lr_decay", type=float, default=0.995, help="")
    parser.add_argument("--lr_decay_step", type=int, default=100, help="")
    parser.add_argument("--lr_decay_step_critic", type=int, default=2000, help="")
    parser.add_argument("--layers", type=str, default="[256, 128]", help="")
    parser.add_argument("--n_embedding", type=int, default=32, help="")
    parser.add_argument("--graph_embedding_size", type=int, default=64, help="")
    parser.add_argument("--n_hidden", type=int, default=24, help="")
    parser.add_argument("--ppo", type=bool, default=False, help="")
    parser.add_argument("--entropy_weight", type=float, default=0.00001, help="")
    parser.add_argument("--reward_scaler", type=float, default=15, help="")
    parser.add_argument("--n_multi_head", type=int, default=2, help="")
    parser.add_argument("--k_hop", type=int, default=2, help="")
    parser.add_argument("--dot_product", type=bool, default=True, help="")
    parser.add_argument("--interval", type=int, default=300, help="")
    parser.add_argument("--fully_connected", type=bool, default=False, help="")
    parser.add_argument("--gnn_type", type=str, default='FastGTN', help="")
    return parser.parse_args()


# def get_cfg():
#     parser = argparse.ArgumentParser(description="")
#     parser.add_argument("--vessl", type=bool, default=False, help="vessl AI 사용여부")
#     parser.add_argument("--step", type=int, default=400001, help="")
#     parser.add_argument("--log_step", type=int, default=10, help="")
#
#     parser.add_argument("--save_step", type=int, default=100000, help="")
#     parser.add_argument("--batch_size", type=int, default=12, help="")
#     parser.add_argument("--C", type=float, default=10, help="")
#     parser.add_argument("--T", type=float, default=1.0, help="")
#     parser.add_argument("--iteration", type=int, default=2, help="")
#     parser.add_argument("--epsilon", type=float, default=0.2, help="")
#     parser.add_argument("--n_glimpse", type=int, default=2, help="")
#     parser.add_argument("--n_process", type=int, default=3, help="")
#     parser.add_argument("--lr", type=float, default=1.e-4, help="")
#     parser.add_argument("--lr_critic", type=float, default=4.e-4, help="")
#     parser.add_argument("--lr_decay", type=float, default=0.98, help="")
#     parser.add_argument("--lr_decay_step", type=int, default=30000, help="")
#     parser.add_argument("--lr_decay_step_critic", type=int, default=200, help="")
#     parser.add_argument("--layers", type=str, default="[512, 256]", help="")
#     parser.add_argument("--n_embedding", type=int, default=32, help="")
#     parser.add_argument("--graph_embedding_size", type=int, default=196, help="")
#     parser.add_argument("--n_hidden", type=int, default=512, help="")
#     return parser.parse_args()