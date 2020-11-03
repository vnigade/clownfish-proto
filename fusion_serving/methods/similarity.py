import torch
from torch import nn
from torch.functional import F
import numpy as np

sim_model = None


def cosine_similarity(prev_vec, cur_vec):
    vec1 = np.array(prev_vec)
    vec2 = np.array(cur_vec)
    dot_prod = vec1 * vec2
    numerator = np.sum(dot_prod)
    vec1_square_root = np.sqrt(np.sum(vec1 ** 2))
    vec2_square_root = np.sqrt(np.sum(vec2 ** 2))
    denominator = vec1_square_root * vec2_square_root

    return (numerator / denominator)


def siminet_similarity(prev_vec, cur_vec):
    assert sim_model is not None, "SimModel cannot be none"
    siminet_input1 = torch.tensor(prev_vec).float()
    siminet_input2 = torch.tensor(cur_vec).float()
    inputs = torch.cat([siminet_input1, siminet_input2], dim=-1)
    outputs = sim_model(inputs)
    outputs = torch.sigmoid(outputs)
    siminet_alpha = outputs.cpu().detach().numpy()[0]
    return siminet_alpha.item()


class SimiNet(nn.Module):
    def __init__(self, n_classes):
        super(SimiNet, self).__init__()
        self.output = 1
        self.input = n_classes * 2
        N_HIDDEN = 1024
        self.l1 = nn.Linear(self.input, N_HIDDEN, bias=True)
        self.l2 = nn.Linear(N_HIDDEN, N_HIDDEN, bias=True)
        self.l3 = nn.Linear(N_HIDDEN, self.output, bias=True)

    def forward(self, x):
        x = x.view(-1, self.input)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def load_siminet_model(n_classes, resume_path):
    model = SimiNet(n_classes)
    # if not opt.no_cuda:
    #    model = model.cuda()
    model = nn.DataParallel(model, device_ids=None)

    print('loading checkpoint {}'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
    res = [val for key, val in checkpoint['state_dict'].items()
           if 'module' in key]
    if len(res) == 0:
        # Model wrapped around DataParallel but checkpoints are not
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    global sim_model
    sim_model = model
    return model


class Similarity:
    def __init__(self, type, opt):

        self.type = type
        if self.type == 'siminet':
            load_siminet_model(opt.n_classes, opt.siminet_path)

        self.rhos = {0: 0.0}

    def reset(self):
        self.rhos = {0: 0.0}

    def get_similarity(self, id, **kwargs):
        if id in self.rhos:
            return self.rhos[id]

        assert len(kwargs) > 0
        rho = 0.5
        if self.type == "cosine":
            rho = cosine_similarity(
                prev_vec=kwargs['vec1_softmax'], cur_vec=kwargs['vec2_softmax'])
        elif self.type == "siminet":
            rho = siminet_similarity(
                prev_vec=kwargs['vec1'], cur_vec=kwargs['vec2'])

        self.rhos[id] = rho
        return rho
