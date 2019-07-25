# ANCHOR Libraries Import
import os
import pickle
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import autograd.numpy as np
import sys
import shutil
import math
import random
import pickle
from tqdm import tqdm
import data
import model
from scipy.linalg import subspace_angles
import math
import torch
from pyfiglet import Figlet

#ANCHOR  Clear terminal and suppress warnings
os.system('clear')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

print("===============================================================================================")
f = Figlet(font='thin')
print f.renderText('Invariance in lottery ticket hypothesis')
print(":::: Code by Adepu Ravi Shankar & Rahul-Vigneswaran K 2019 ::::")
print("===============================================================================================\n")

# ANCHOR Initialize Variables

prev_hess = 0
prev_eigval = 0
prev_eigvec = 0
initial_model = []

# ANCHOR Parser
parser = argparse.ArgumentParser()
# Pruning
parser.add_argument('--per', type=float, default=10.,
                    help='Pruning Percentage')
parser.add_argument('--prune_iter', type=int, default=10,
                    help='Decided the no. of times the pruning has to be done.')

# Hessian
parser.add_argument('--top', type=int, default=100,
                    help='Dimension of the top eigenspace')
parser.add_argument('--suffix', type=str, default='new',
                    help='suffix to save npy array')
parser.add_argument('--freq', type=int, default='1',
                    help='freq of Hess calculation')

# Data Generation
parser.add_argument('--data-type', type=str, choices=['blob', 'circle', 'moon'], default='blob',
                    help='Type of random data generation pattern.')
parser.add_argument('--num-samples', type=int, default=1000,
                    help='Number of training samples')
parser.add_argument('--input-dim', type=int, default=5,
                    help='Dimension of the input space')
parser.add_argument('--num-classes', type=int, default=2,
                    help='Number of classes in generated data. '
                         'Taken into account only by classifiers and data generators that support multi-class')
parser.add_argument('--cov-factor', type=float, default=1.,
                    help='Multiplier for the covariance of the data')
parser.add_argument('--data-seed', type=int, default=None,
                    help='Seed for random number generation of data generators')

# Model
parser.add_argument('--classifier', type=str, choices=['logreg', 'fullconn'], default='fullconn',
                    help='Type of classifier. Logistic Regression, Fully-Connected NN.')
parser.add_argument('--layer-sizes', nargs='*', type=int, default=[10, 5],
                    help='Number of units in hidden layers. '
                         'First layer will have --input-dim units. Last layer will have --num-classes units.')

# Training
parser.add_argument('--batch-size', type=int, default=0,
                    help='Number of samples in a training batch. '
                         '0 means whole dataset')
parser.add_argument('--learning-rate', type=float, default=0.01,
                    help='Learning rate')
parser.add_argument('--stopping-grad-norm', type=float, default=1e-4,
                    help='Stop training if grad_norm becomes smaller than this threshold')
parser.add_argument('--max-iterations', type=int, default=100,
                    help='Cancel training after maximum number of iteration steps')

# Results
parser.add_argument('--hessian-calc-period', type=int, default=1,
                    help='Calculate hessian at every N iteration. '
                         '0 means do not calculate at intermediate iterations.')
parser.add_argument('--results-folder', type=str, default='/home/ravi/eclipse-workspace/saguant/hessian-for-basicDL/hessianexps/results',
                    help='Folder in which to put all results folders')
parser.add_argument('--experiment-folder', type=str, default='defaults',
                    help='Folder in which to write results files')

# Hessian analysis
parser.add_argument('--top-evals', type=int, default=100,
                    help='Find top k evals')
parser.add_argument('--bottom-evals', type=int, default=0,
                    help='Find bottom k evals')

args = parser.parse_args()

# ANCHOR Yes or no directory definition


def yes_or_no():
    while True:
        yes = {'yes', 'y', 'ye', ''}
        no = {'no', 'n'}

        choice = raw_input(
            "Do you want me to delete the directory? (y/n) \n\n").lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            sys.stdout.write("\nPlease respond with 'yes' or 'no' \n\n")

# ANCHOR Yes or no plot definition

def yes_or_no_image():
    while True:
        yes = {'yes', 'y', 'ye', ''}
        no = {'no', 'n'}

        choice = raw_input(
            "Do you want to see the plot? (y/n) \n\n").lower()
        if choice in yes:
            return True
        elif choice in no:
            return False
        else:
            sys.stdout.write("\nPlease respond with 'yes' or 'no' \n\n")


# ANCHOR Generate Results folder
args.results_folder = os.getcwd() + '/' + 'results/'+'B_size-'+str(args.batch_size)+'-Arch-'+str(args.input_dim)+str(args.layer_sizes)+'-iters-'+str(args.max_iterations) + \
    '-data-'+str(args.data_type)+'-hess_freq-'+str(args.hessian_calc_period)+'-top-' + \
    str(args.top)+'--freq-'+str(args.freq)+'--iter-'+str(args.max_iterations)+'--prune_iter'+str(args.prune_iter)+'--prune_per'+str(args.per)
if not os.path.exists(args.results_folder):
    os.mkdir(args.results_folder)
else:
    print("\nDirectory already exists !!\n")
    a1 = yes_or_no()
    if a1 == True:
        # Prompt to delete directory if it already exists
        shutil.rmtree(args.results_folder, ignore_errors=True)
        print("====> Directory Deleted")
        os.mkdir(args.results_folder)
        print("====> Directory recreated")
        print("\n")
        print("===============================================================================================\n")


    else:
        print("Directory Already exists and was not opted for deletion.\n\n")
        sys.exit()

if args.classifier == 'logreg' and args.data_type == 'blob' and args.num_classes != 2:
    raise Exception('LogReg for more than 2 classes is not implemented yet.')


# ANCHOR Main
def main():
    coeff = []
    ang_sb = []
    ang_np = []
    p_angles = []
    inputs_train, targets_train, inputs_test, targets_test = data.generate_data(
        args)
    results = {
        'inputs_train': inputs_train,
        'targets_train': targets_train,
        'inputs_test': inputs_test,
        'targets_test': targets_test
    }

    # Actual Model that is being observed
    mdl = model.create_model(args, inputs_train, targets_train)
    print("\n===============================================================================================\n")

    start_params = mdl.params_flat

    # NOTE Pickling Initial Weights
    with open('outfile', 'wb') as sp:
        pickle.dump(mdl.params_flat, sp)

    new_params = train_model(args, mdl, results)

    with open('outfile', 'rb') as sp:
        start_params = pickle.load(sp)

    # NOTE Lottery Ticket Pruning Loop
    per = args.per
    nonzer = (np.count_nonzero(mdl.params_flat))
    zer = len(mdl.params_flat) - nonzer
    x1 = nonzer-zer
    z1 = int(((x1/100.)*per))
    zer = z1 + zer
    print(" {} + {} = {}".format(0, nonzer, len(mdl.params_flat)))

    new_params, inputs, outputs = train_model(args, mdl, results)
    hess = mdl.hessian(mdl.params_flat)     # Calculating Hessian
    # Converting the Hessian to Tensor
    hess = torch.tensor(hess).float()
    eigenvalues, eigenvec = torch.symeig(hess, eigenvectors=True)

    hess, eigenvalues, eigenvec, coeff, ang_np, ang_sb, p_angles, top_vec = invar(
        mdl, args, inputs_train, targets_train, hess, eigenvalues, eigenvec, coeff, ang_np, ang_sb, p_angles)

    # NOTE Pruning Loop

    print("===============================================================================================\n")
    for i in tqdm(range(0, args.prune_iter), desc="Pruning Progress", dynamic_ncols=True):
        print("\n{} +".format(zer)),

        pruned_params_flat, zer, nonzer = prune_function(mdl, zer)
        print("{} = {}".format(nonzer, len(mdl.params_flat)))
        x1 = nonzer-zer
        z1 = int((x1/100.)*per)
        zer = z1 + zer

        for p in range(0, len(start_params)):
            if (pruned_params_flat[p] != 0.):
                pruned_params_flat[p] = start_params[p]

        mdl.params_flat = pruned_params_flat
        new_params, coeff = train_pruned_model(
            args, mdl, results, top_vec, coeff)

    coeff = torch.tensor(coeff)
    for i in range(coeff.shape[0]):
        a = torch.zeros(coeff[i].shape[0]).long()
        b = torch.arange(0, coeff[i].shape[0])
        c = torch.where(((coeff[i] > -0.1) & (coeff[i] < 0.1)), b, a)
        z = torch.zeros(coeff[i].shape[0]).fill_(0)
        z[torch.nonzero(c)] = coeff[i][torch.nonzero(c)]
        z = np.array(z)
        plt.plot(z)
    plt.xlabel('Dimension')
    plt.ylabel('Coefficient')
    pnpy = args.results_folder+'/plot1'+'.png'
    plt.savefig(pnpy, format='png')

    args.suffix = args.results_folder+'/coeff.npy'
    np.save(args.suffix, coeff)
    args.suffix = args.results_folder+'/ang_sb.npy'
    np.save(args.suffix, ang_sb)
    args.suffix = args.results_folder+'/ang_np.npy'
    np.save(args.suffix, ang_np)
    args.suffix = args.results_folder+'/p_angles.npy'
    np.save(args.suffix, p_angles)

    return args.results_folder
    


# ANCHOR Definition for Single Pruning
def prune_function(mdl, j):
    zer = 0
    b = mdl.params_flat
    while zer <= (j):
        i = random.randrange(0, len(b))
        b[i] = 0.
        nonzer = (np.count_nonzero(b))
        zer = len(b) - nonzer
    return b, zer, nonzer


# ANCHOR Invariance defintion for Non pruned network
def invar(mdl, args, inputs_train, targets_train, prev_hess, prev_eigval, prev_eigvec, coeff, ang_np, ang_sb, p_angles):
    # calculating hessian
    hess = mdl.hessian(mdl.params_flat)                 # Calculating Hessian
    # Converting the Hessian to Tensor
    hess = torch.tensor(hess).float()
    # Extracting the eigenvalues and Eigen Vectors from the Calculated Hessian
    eigenvalues, eigenvec = torch.symeig(hess, eigenvectors=True)

    top = args.top      # This decides how many top eigenvectors are considered
                                # |The reason for negative top :: torch.symeig outputs eigen vectors in the increasing order and as a result |
    dom = eigenvec[:, -top:]
                                # |       mdl                        the top (maximum) eigenvectors will be atlast.                             |
    dom = dom.float()
    # A random vector which is of the dim of variable "top" is being initialized
    alpha = torch.rand(top)

    # Finding the top vector
    # Representing alpha onto dominant eigen vector
    vec = (alpha*dom.float()).sum(1)
    vec = vec/torch.sqrt((vec*vec).sum())     # Normalization of top vector

    # Dummy Model for calculating gradient
    mdl_test = model.create_model(args, inputs_train, targets_train)

    # Finding gradient at top vec using Dummy network.
    mdl_test.params_flat = np.array(vec)

    # Find coeff and append.
    top_vec = mdl_test.params_flat
    c = torch.mv(hess.transpose(0, 1), torch.tensor(
        mdl_test.params_flat).float())
    if np.size(coeff) == 0:
        coeff = c.detach().cpu().numpy()
        coeff = np.expand_dims(coeff, axis=0)
    else:
        coeff = np.concatenate(
            (coeff, np.expand_dims(c.detach().cpu().numpy(), axis=0)), 0)

    # Statistics of subspaces, (1) Angle between top subpaces
    eigenvalues_prev, eigenvec_prev = torch.symeig(
        prev_hess, eigenvectors=True)
    # Is it not the same as the variable "dom" that was calculated earlier ?
    dom_prev = eigenvec_prev[:, -top:]

    # calculation 1 norm, which is nothing but angle between subspaces
    ang = np.linalg.norm(torch.mm(dom_prev, dom.transpose(0, 1)).numpy(), 1)
    ang_sb.append(ang)
    ang = np.rad2deg(subspace_angles(dom_prev, dom))
    ang_np.append(ang)

    # Calculating principal angles
    u, s, v = torch.svd(torch.mm(dom.transpose(0, 1), dom_prev))

    # Output in radians
    s = torch.acos(torch.clamp(s, min=-1, max=1))
    s = s*180/math.pi

    # Attach 's' to p_angles
    if np.size(p_angles) == 0:
        p_angles = s.detach().cpu().numpy()
        p_angles = np.expand_dims(p_angles, axis=0)
    else:
        p_angles = np.concatenate(
            (p_angles, np.expand_dims(s.detach().cpu().numpy(), axis=0)), 0)
    prev_hess = hess
    prev_eigval = eigenvalues
    prev_eigvec = eigenvec

    return hess, eigenvalues, eigenvec, coeff, ang_np, ang_sb, p_angles, top_vec


# ANCHOR Train Unpruned Definition
def train_model(args, mdl, results):

    all_w = []
    results['args'] = args
    init_loss = mdl.loss(mdl.params_flat)
    init_grad_norm = np.linalg.norm(mdl.gradient(mdl.params_flat))

    print('Initial loss: {}, norm grad: {}'.format(init_loss, init_grad_norm))
    results['init_full_loss'] = init_loss
    results['init_full_grad_norm'] = init_grad_norm

    results['history1'] = []
    results['history1_columns'] = ['iter_no', 'batch_loss',
                                   'batch_grad_norm', 'batch_param_norm']
    results['history2'] = []
    results['history2_columns'] = ['full_hessian', 'full_hessian_evals']

    for iter_no in tqdm(range(args.max_iterations)):
        inputs, targets = get_batch_samples(iter_no, args, mdl)
        batch_loss = mdl.loss(mdl.params_flat, inputs, targets)
        batch_grad = mdl.gradient(mdl.params_flat, inputs, targets)
        batch_grad_norm = np.linalg.norm(batch_grad)
        batch_param_norm = np.linalg.norm(mdl.params_flat)


#    saving weights in all iterations
        if batch_grad_norm <= args.stopping_grad_norm:
            break
        mdl.params_flat -= batch_grad * args.learning_rate
        
        all_w.append(np.power(math.e, mdl.params_flat))
        

    final_loss = mdl.loss(mdl.params_flat)
    final_grad_norm = np.linalg.norm(mdl.gradient(mdl.params_flat))
    print('Final loss: {}, norm grad: {}\n'.format(final_loss, final_grad_norm))

    return mdl.params

# ANCHOR Train Pruned Definition


def train_pruned_model(args, mdl, results, top_vec, coeff):

    all_w = []
    results['args'] = args
    init_loss = mdl.loss(mdl.params_flat)
    init_grad_norm = np.linalg.norm(mdl.gradient(mdl.params_flat))

    print('Initial loss: {}, norm grad: {}'.format(init_loss, init_grad_norm))
    results['init_full_loss'] = init_loss
    results['init_full_grad_norm'] = init_grad_norm

    results['history1'] = []
    results['history1_columns'] = ['iter_no', 'batch_loss',
                                   'batch_grad_norm', 'batch_param_norm']
    results['history2'] = []
    results['history2_columns'] = ['full_hessian', 'full_hessian_evals']

    for iter_no in tqdm(range(args.max_iterations)):
        inputs, targets = get_batch_samples(iter_no, args, mdl)
        batch_loss = mdl.loss(mdl.params_flat, inputs, targets)
        batch_grad = mdl.gradient(mdl.params_flat, inputs, targets)
        batch_grad_norm = np.linalg.norm(batch_grad)
        batch_param_norm = np.linalg.norm(mdl.params_flat)

        if iter_no % args.freq == 0:

            # calculating hessian
            # Calculating Hessian
            hess = mdl.hessian(mdl.params_flat)
            # Converting the Hessian to Tensor
            hess = torch.tensor(hess).float()
            c = torch.mv(hess.transpose(0, 1), torch.tensor(top_vec).float())
            if np.size(coeff) == 0:
                coeff = c.detach().cpu().numpy()
                coeff = np.expand_dims(coeff, axis=0)
            else:
                coeff = np.concatenate(
                    (coeff, np.expand_dims(c.detach().cpu().numpy(), axis=0)), 0)


#    saving weights in all iterations
        if batch_grad_norm <= args.stopping_grad_norm:
            break
        mdl.params_flat -= batch_grad * args.learning_rate
        all_w.append(np.power(math.e, mdl.params_flat))
        
    final_loss = mdl.loss(mdl.params_flat)
    final_grad_norm = np.linalg.norm(mdl.gradient(mdl.params_flat))
    print('Final loss: {}, norm grad: {}\n'.format(final_loss, final_grad_norm))

    return mdl.params, coeff


# ANCHOR Batch Sample Definition
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


if __name__ == '__main__':

    args.results_folder = main()
    print("\n\n The code has been successfully executed!!\n\n")
    a11 = yes_or_no_image()
    if a11 == True :
        pnpy = args.results_folder+'/plot1'+'.png'
        import cv2 
        img = cv2.imread(pnpy)  
        cv2.imshow('Plot', img)  
        cv2.waitKey(0)         
        cv2.destroyAllWindows() 
        print("\n\n Tata!!\n\n")
    else:
        print("\n\n Tata!!\n\n")

    print("\n===============================================================================================\n")

