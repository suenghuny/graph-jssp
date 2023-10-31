import argparse

def get_cfg():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--vessl", type=bool, default=False, help="vessl AI 사용여부")
    parser.add_argument("--step", type=int, default=400001, help="")
    parser.add_argument("--log_step", type=int, default=10, help="")

    parser.add_argument("--save_step", type=int, default=100000, help="")
    parser.add_argument("--batch_size", type=int, default=12, help="")
    parser.add_argument("--C", type=float, default=1, help="")
    parser.add_argument("--T", type=int, default=1, help="")
    parser.add_argument("--iteration", type=int, default=1, help="")
    parser.add_argument("--epsilon", type=float, default=0.18, help="")
    parser.add_argument("--n_glimpse", type=int, default=2, help="")
    parser.add_argument("--n_process", type=int, default=3, help="")
    parser.add_argument("--lr", type=float, default=1.e-4, help="")
    parser.add_argument("--lr_critic", type=float, default=1.e-3, help="")
    parser.add_argument("--lr_decay", type=float, default=0.98, help="")
    parser.add_argument("--lr_decay_step", type=int, default=30000, help="")
    parser.add_argument("--lr_decay_step_critic", type=int, default=200, help="")
    parser.add_argument("--layers", type=str, default="[196, 108]", help="")
    parser.add_argument("--n_embedding", type=int, default=32, help="")
    parser.add_argument("--graph_embedding_size", type=int, default=24, help="")
    parser.add_argument("--n_hidden", type=int, default=64, help="")
    return parser.parse_args()