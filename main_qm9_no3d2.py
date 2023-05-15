# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
from configs.datasets_config import get_dataset_info
import os
from os.path import join, exists
from qm9 import dataset
from qm9.models import get_optim, get_model, get_prop_dist
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from qm9.utils import prepare_context, compute_mean_mad
from train_test_no3d import train_epoch, test, analyze_and_save

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

parser = argparse.ArgumentParser(description='TransformerDiffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')

# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=5)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=50)
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=10,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')



# <-- Global encoding args
parser.add_argument('--multi_hop_max_dist', type=int, default=2,
                    help='')
parser.add_argument('--num_encoder_layers', type=int, default=6,
                    help='')
parser.add_argument('--embedding_dim', type=int, default=128,
                    help='')
parser.add_argument('--edge_embedding_dim', type=int, default=128,
                    help='')
parser.add_argument('--graph_embedding_dim', type=int, default=32,
                    help='')
parser.add_argument('--num_attention_heads', type=int, default=8,
                    help='')
parser.add_argument('--num_3d_bias_kernel', type=int, default=16,
                    help='')


parser.add_argument('--use_2d_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--use_3d_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--use_2d_neighbor_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--use_3d_neighbor_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--apply_concrete_adjacency_neighbor', type=bool, default=False, 
                    help='')
parser.add_argument('--use_2d_edge_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--trainable_dist_proj', type=bool, default=True, 
                    help='')
parser.add_argument('--use_extra_graph_embedding', type=bool, default=False, 
                    help='')
parser.add_argument('--use_extra_graph_embedding_attn_bias', type=bool, default=False, 
                    help='')


parser.add_argument('--cutoff_upper', type=float, default=4.0,
                    help='')
parser.add_argument('--cutoff_lower', type=float, default=0.0,
                    help='')


parser.add_argument('--use_edge_type', type=str, default='no',
                    help='no, multi_hop')
parser.add_argument('--distance_projection', type=str, default='exp',
                    help='exp, gaussian')
parser.add_argument('--neighbor_combine_embedding', type=str, default='cat',
                    help='cat, add, no')
parser.add_argument('--extra_feature_type', type=str, default='all',
                    help='all, cycles, eigenvalues')



# Transformer args
parser.add_argument('--ffn_embedding_dim', type=int, default=300,
                    help='')
parser.add_argument('--ffn_edge_embedding_dim', type=int, default=300,
                    help='')
parser.add_argument('--ffn_graph_embedding_dim', type=int, default=100,
                    help='')
parser.add_argument('--before_attention_qn_block_size', type=int, default=0,
                    help='')
parser.add_argument('--in_attention_qn_block_size', type=int, default=0,
                    help='')


parser.add_argument('--before_attention_dropout', type=float, default=0,
                    help='')
parser.add_argument('--before_attention_quant_noise', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_feature_dropout', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_dropout', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_activation_dropout', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_activation_dropout_adj', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_activation_dropout_graph_feature', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_quant_noise', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_droppath', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_droppath_adj', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_droppath_graph_feature', type=float, default=0,
                    help='')


parser.add_argument('--before_attention_layernorm', type=bool, default=True, 
                    help='')
parser.add_argument('--in_attention_layernorm', type=bool, default=True, 
                    help='')
parser.add_argument('--in_attention_pred_adjacency', type=bool, default=True, 
                    help='')


parser.add_argument('--attention_activation_fn', type=str, default='silu',
                    help='silu, relu, gelu, softmax')



# Equivariant Transformer args
parser.add_argument('--use_equivariant_transformer', type=bool, default=True, 
                    help='')
parser.add_argument('--equivariant_use_x_layernorm', type=bool, default=True, 
                    help='')
parser.add_argument('--equivariant_use_dx_layernorm', type=bool, default=True, 
                    help='')
parser.add_argument('--equivariant_apply_concrete_adjacency', type=bool, default=True, 
                    help='')


parser.add_argument('--equivariant_in_attention_dropout', type=float, default=0,
                    help='')
parser.add_argument('--equivariant_dx_dropout', type=float, default=0,
                    help='')


parser.add_argument('--equivariant_attention_activation_fn', type=str, default='silu',
                    help='silu, relu, gelu, softmax')
parser.add_argument('--equivariant_distance_influence', type=str, default='both',
                    help='both, keys, values')


# Output args
parser.add_argument('--combine_transformer_output', type=str, default='cat',
                    help='cat, add')


parser.add_argument('--use_output_projection', type=bool, default=True, 
                    help='')
parser.add_argument('--use_equivariant_output_projection', type=bool, default=False, 
                    help='')

args = parser.parse_args()

dataset_info = get_dataset_info(args.dataset, args.remove_h)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print('On device:', device, torch.cuda.is_available())
#device = torch.device("cuda:0")
dtype = torch.float32

bs = args.batch_size
if exists(join('outputs', args.exp_name, 'args.pickle')) and (args.resume is None):
    with open(join('outputs', args.exp_name, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
        
    args.break_train_epoch = False
    args.batch_size = bs
    print(args)


if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr

    print(args)


utils.create_folders(args)
# print(args)



if len(args.conditioning) > 0:
    print(f'Conditioning on {args.conditioning}')
    property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
    context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
    context_node_nf = context_dummy.size(2)
else:
    context_node_nf = 0
    property_norms = None

args.context_node_nf = context_node_nf


gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.

def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def main(local_rank):
    model, nodes_dist = get_model(args, dataset_info)

    if exists(join('outputs', args.exp_name, 'generative_model.npy')) and exists(join('outputs', args.exp_name, 'optim.npy')) and (args.resume is None):
        print('Resume training for', join('outputs', args.exp_name, 'generative_model.npy'))
        flow_state_dict = torch.load(join('outputs', args.exp_name, 'generative_model.npy'))
        model.load_state_dict(flow_state_dict)
        print('Done loading for', join('outputs', args.exp_name, 'generative_model.npy'))


    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model.npy'))
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)


    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:

        ip = os.environ['MASTER_IP']
        port = os.environ['MASTER_PORT']
        hosts = int(os.environ['WORLD_SIZE']) 
        rank = int(os.environ['RANK']) 
        gpus = torch.cuda.device_count()

        dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts*gpus, rank=rank*gpus+local_rank)
        torch.cuda.set_device(local_rank)
        model.cuda(local_rank)
        #model_dp = torch.nn.DataParallel(model.cpu())
        #model_dp = model_dp.cuda()
        model_dp = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    else:
        rank = 0
        model = model.to(device)
        model_dp = model


    optim = get_optim(args, model)
    if exists(join('outputs', args.exp_name, 'optim.npy')) and (args.resume is None):
        optim_state_dict = torch.load(join('outputs', args.exp_name, 'optim.npy'))
        optim.load_state_dict(optim_state_dict)
        

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            #model_ema_dp = torch.nn.DataParallel(model_ema)
            model_ema_dp = DistributedDataParallel(model_ema, device_ids=[local_rank], find_unused_parameters=True)
        
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    
    # Retrieve QM9 dataloaders
    dataloaders, charge_scale, train_sampler = dataset.retrieve_dataloaders(args, use3d=False)

    data_dummy = next(iter(dataloaders['train']))
    prop_dist = get_prop_dist(args, dataloaders['train'])

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
        
        

    best_nll_val = 1e8
    best_nll_test = 1e8
    for epoch in range(args.start_epoch, args.n_epochs):
        if (train_sampler is not None) and (torch.cuda.device_count() > 1):
            train_sampler.set_epoch(epoch)
            
        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, rank=rank, local_rank=local_rank)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0:

            if not args.break_train_epoch:
                analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=args.n_stability_samples)
                
            nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms, rank=rank, local_rank=local_rank)
            
            nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms, rank=rank, local_rank=local_rank)

            if rank == 0 and local_rank == 0:
                if nll_val < best_nll_val:
                    best_nll_val = nll_val
                    best_nll_test = nll_test

                    if args.save_model:
                        args.current_epoch = epoch + 1
                        utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                        utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                        if args.ema_decay > 0:
                            utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                        with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                            pickle.dump(args, f)
                            
                print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
                print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()

    if args.dp and ngpus > 1:
        print(f'Training using {ngpus} GPUs')
        torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
    else:
        main(0)
