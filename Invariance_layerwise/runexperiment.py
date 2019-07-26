import os
import pickle
from datetime import datetime

import autograd.numpy as np

import config
import data
import model
from rayleigh import get_top_eigensystem, get_bottom_eigensystem

import torch

prev_hess = 0

def main():
    args = config.parse_command_line_arguments()

    inputs_train, targets_train, inputs_test, targets_test = data.generate_data(args)
    results = {
        'inputs_train': inputs_train,
        'targets_train': targets_train,
        'inputs_test': inputs_test,
        'targets_test': targets_test
    }

    mdl = model.create_model(args, inputs_train, targets_train)

    train_model(args, mdl, results)


def train_model(args, mdl, results):
    coeff = []
    results['args'] = args
    init_loss = mdl.loss(mdl.params_flat)
    init_grad_norm = np.linalg.norm(mdl.gradient(mdl.params_flat))
    print('Initial loss: {}, norm grad: {}'.format(init_loss, init_grad_norm))
    results['init_full_loss'] = init_loss
    results['init_full_grad_norm'] = init_grad_norm

    results['history1'] = []
    results['history1_columns'] = ['iter_no', 'batch_loss', 'batch_grad_norm', 'batch_param_norm']
    results['history2'] = []
    results['history2_columns'] = ['full_hessian', 'full_hessian_evals']
    for iter_no in range(args.max_iterations):
        inputs, targets = get_batch_samples(iter_no, args, mdl)
        batch_loss = mdl.loss(mdl.params_flat, inputs, targets)
        batch_grad = mdl.gradient(mdl.params_flat, inputs, targets)
        batch_grad_norm = np.linalg.norm(batch_grad)
        batch_param_norm = np.linalg.norm(mdl.params_flat)

        if iter_no % 10 == 0:
            hess = mdl.hessian(mdl.params_flat)
        if iter_no % 10 == 1:          
            c = analyse_hessian(args, mdl, hess, batch_grad)
            if np.size(coeff) == 0:
                coeff = c.detach().cpu().numpy()
                coeff = np.expand_dims(coeff, axis=0)
            else:
                coeff = np.concatenate((coeff,np.expand_dims(c.detach().cpu().numpy(),axis=0)),0)                
            evals = np.linalg.eigvalsh(hess)

        if batch_grad_norm <= args.stopping_grad_norm:
            break

        mdl.params_flat -= batch_grad * args.learning_rate
        print('{:06d} {} loss: {:.8f}, norm grad: {:.8f}'.format(
            iter_no, datetime.now(), batch_loss, batch_grad_norm))

    final_loss = mdl.loss(mdl.params_flat)
    final_grad_norm = np.linalg.norm(mdl.gradient(mdl.params_flat))
    print('Final loss: {}, norm grad: {}'.format(final_loss, final_grad_norm))
    np.save(args.suffix,coeff)

def analyse_hessian(args,mdl, hess, batch_grad):
    hess = torch.tensor(hess).float()
    prev_hess = hess
    eigenvalues, eigenvec = torch.symeig(hess,eigenvectors=True)
    top = args.top
    dom = eigenvec[:,-top:]
    dom = dom.float()
    
    alpha=torch.rand(top)
    vec=(alpha*dom.float()).sum(1)
    vec=vec/torch.sqrt((vec*vec).sum())
    w_t = mdl.params_flat - vec * 0.0001

    dom = eigenvec.float()
        
    # Since inv_h is a 570x20 dim we need to append 0's in all last columns
    if dom.shape[0] - dom.shape[1] != 0:
        dom = torch.cat((dom, torch.FloatTensor(dom.shape[0], dom.shape[0]- dom.shape[1]).fill_(0)),1)
    coeff =  torch.mv(prev_hess.transpose(0,1), torch.tensor(w_t).float())
    return coeff


def get_batch_samples(iter_no, args, mdl):
    """Return inputs and outputs belonging to batch given iteration number."""
    if args.batch_size == 0:
        return None, None

    num_batches = int(np.ceil(len(mdl.inputs) / args.batch_size))
    mod_iter_no = iter_no % num_batches
    start = mod_iter_no * args.batch_size
    end = (mod_iter_no + 1) * args.batch_size
    inputs = mdl.inputs[start:end]
    targets = mdl.targets[start:end]
    return inputs, targets


def save_results(args, results):
    if not os.path.exists(args.results_folder):
        os.makedirs(args.results_folder)
    now = datetime.now()
    now_str = now.strftime('%Y%m%d-%H%M%S-%f')
    filename = 'results-{}.pkl'.format(now_str)
    filepath = os.path.join(args.results_folder, filename)
    with open(filepath, 'wb') as fh:
        pickle.dump(results, fh)


def compute_eigensystem(args, mdl):
    """Compute the eigenvalues of the symmetric (exact) Hessian."""
    evs = np.linalg.eigvalsh(mdl.hessian(mdl.params_flat))
    print('all eigenvalues {}'.format(np.array(sorted(evs))))

    top_evals, top_evecs = get_top_eigensystem(mdl, args.input_dim, args.top_evals)
    bottom_evals, bottom_evecs = get_bottom_eigensystem(mdl, args.input_dim, args.bottom_evals)
    print('largest eigenvalue by Raleigh quotient {}'.format(np.array(sorted(top_evals))))
    print('smallest eigenvalue by Raleigh quotient {}'.format(np.array(sorted(bottom_evals))))

    srayeigs = np.array(sorted(bottom_evals) + sorted(top_evals))
    seigs = np.array(sorted(evs[:args.bottom_evals]) + sorted(evs[-args.top_evals:]))
    print('percentage diff', np.array((srayeigs - seigs) / seigs * 100., dtype=np.int32))


if __name__ == '__main__':
    main()
