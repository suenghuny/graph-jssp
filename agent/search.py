import os
import torch

from actor import PtrNet1


def sampling(env, params, batch_size, test_input):
    test_inputs_temp = test_input.repeat(batch_size, 1, 1)
    if env.distribution == "lognormal":
        test_inputs = test_inputs_temp / test_inputs_temp.amax(dim=(1, 2)).unsqueeze(-1).unsqueeze(-1). \
            expand(-1, test_inputs_temp.shape[1], test_inputs_temp.shape[2])
    elif env.distribution == "uniform":
        test_inputs = test_inputs_temp / 100
    # device = torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = PtrNet1(params).to(device)
    if os.path.exists(params["model_path"]):
        checkpoint = torch.load(params["model_path"], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict_actor'])
        model.eval()
    else:
        print('specify pretrained model path')

    pred_sequence, _, _ = model(test_inputs, device)
    makespan_batch = env.stack_makespan(test_inputs_temp, pred_sequence)
    index_makespan_min = torch.argmin(makespan_batch)
    best_sequence = pred_sequence[index_makespan_min]
    best_makespan = makespan_batch[index_makespan_min]
    return best_sequence, best_makespan