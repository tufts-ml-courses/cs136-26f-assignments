import argparse
import numpy as np
import pandas as pd
import os

import torch
import torch.optim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from collections import OrderedDict
from torchvision import datasets, transforms
from torchvision.utils import save_image

from ca3_vae import VariationalAutoencoder

def eval_model_on_data(
        model, dataset_nickname, data_loader, args):
    ''' Evaluate an encoder/decoder model on a dataset

    Returns
    -------
    nelbo_loss : float
    '''
    model.eval()
    total_nelbo_loss = 0.0
    n_seen = 0
    total_1pix = 0.0
    for batch_idx, (batch_data, _) in enumerate(data_loader):
        batch_x_ND = batch_data.view(-1, model.n_dims_data)
        total_1pix += torch.sum(batch_x_ND)
        nelbo = model.calc_nelbo(batch_x_ND, do_rescale=False)

        total_nelbo_loss += nelbo.item()
        n_seen += batch_x_ND.shape[0]
    msg = "%s data: %d images. Total pixels on: %d. Frac pixels on: %.3f" % (
        dataset_nickname, n_seen, total_1pix, total_1pix / float(n_seen*model.n_dims_data))

    nelbo_per_image = total_nelbo_loss / float(n_seen)
    return float(nelbo_per_image), msg

def encode_data_by_digit_category(model, data_loader, n_per_category=1000):
    ''' Encodes a dataset, grouping the latent codes by category

    Returns
    -------
    z_AC_by_cat : OrderedDict with up to n_per_category latent codes per digit
    '''

    model.eval()
    z_AC_by_cat = OrderedDict()
    for cat in range(10):
        z_AC_by_cat[cat] = np.zeros((0, model.n_dims_latent))

    for batch_idx, (batch_data, batch_y) in enumerate(data_loader):
        # N = num examples per batch
        batch_x_ND = batch_data.view(-1, model.n_dims_data)
        batch_y_N = batch_y.view(-1).detach().numpy()
        batch_z_NC = model.encode(batch_x_ND).detach().numpy()
        for cat in range(10):
            n_cur = z_AC_by_cat[cat].shape[0]
            n_new = np.maximum(0, n_per_category - n_cur)

            batch_z_AC = batch_z_NC[batch_y_N == cat]
            z_AC_by_cat[cat] = np.vstack([
                z_AC_by_cat[cat], batch_z_AC[:n_new]])

    return z_AC_by_cat

def plot_encoding_colored_by_digit_category(z_AC_by_cat, xlims=(-2.0, 2.0)):
    ''' Diagnostic visualization of the encoding space

    Post Condition
    --------------
    Creates visual in a matplotlib figure
    '''
    digit_markers = [
        '$\u0030$',
        '$\u0031$',
        '$\u0032$',
        '$\u0033$',
        '$\u0034$',
        '$\u0035$',
        '$\u0036$',
        '$\u0037$',
        '$\u0038$',
        '$\u0039$',
        ]

    tab10_colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        ]
    tab10_colors = [(float(r)/256., float(g)/256., float(b)/256., 0.1) for (r,g,b) in tab10_colors]
    figh, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(5,5))
    for cat in z_AC_by_cat:
        z_cc_AC = z_AC_by_cat[cat]
        ax[0,0].plot(z_cc_AC[:,0], z_cc_AC[:,1], 
            linestyle='',
            marker=digit_markers[cat], markersize=10, color=tab10_colors[cat])

    handles = [
        Line2D([], [], linestyle='', marker=digit_markers[c], markersize=10,
            color=tab10_colors[c][:3], label=str(c))
        for c in range(10)]
    ax[0,0].legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0),
        frameon=False, handletextpad=0.1, title='digit')

    if xlims == 'auto':
        all_z_AC = np.vstack(list(z_AC_by_cat.values()))
        if all_z_AC.shape[0] > 0:
            B = max(1.1 * np.percentile(np.abs(all_z_AC), 99.5), 1e-3)
        else:
            B = 1.0
        xlims = (-B, B)

    ax[0,0].set_xlim(xlims)
    ax[0,0].set_ylim(xlims)
    return figh


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo of VAE on MNIST')
    # arguments that define the model + its initialization
    parser.add_argument(
        '--latent_dim', type=int, default=2,
        help='latent dimension (default: 2)')
    parser.add_argument(
        '--q_sigma_init', type=float, default=0.05,
        help='Initial stddev of approximate posterior (default: 0.05)')
    
    # optimization arguments
    parser.add_argument(
        '--n_epochs', type=int, default=10,
        help="number of epochs (default: 10)")
    parser.add_argument(
        '--batch_size', type=int, default=100,
        help='batch size (default: 100)')
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate for grad. descent (default: 0.001)')
    
    # reproducibility and file structure arguments
    parser.add_argument(  
        '--seed', type=int, default=8675309,
        help='random seed (default: 8675309)')
    parser.add_argument(
        '--filename_prefix', type=str, default='VAE-latent_dim=$latent_dim-lr=$lr')
    parser.add_argument(
        '--data_folder', type=str, default=os.path.dirname(os.path.abspath(__file__)),
        help='Folder where you have MNIST data stored')
    args = parser.parse_args()

    ## Set random seed
    torch.manual_seed(args.seed)

    ## Set filename_prefix for results
    for key, val in args.__dict__.items():
        args.filename_prefix = args.filename_prefix.replace('$' + key, str(val))
    print("Saving with prefix: %s" % args.filename_prefix)

    S = 24 # crop 2 pixels on each side
    n_dims_data = S**2

    ## Create AE model by calling its constructor
    model = VariationalAutoencoder(
        n_dims_data=n_dims_data,
        n_dims_latent=args.latent_dim,
        q_sigma_init=args.q_sigma_init)

    ## Create generators for grabbing batches of train or test data
    # Each loader will produce **binary** data arrays (using transforms defined below)
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_folder, train=True, download=False,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),    
        batch_size=args.batch_size,
        shuffle=True)

    eval_batch_size = 2000
    train_eval_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_folder, train=True, download=False,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),    
        batch_size=eval_batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_folder, train=False, download=False,
            transform=transforms.Compose([
                transforms.CenterCrop((S)),
                transforms.ToTensor(), torch.round])),
        batch_size=eval_batch_size, shuffle=False)

    print("MNIST train data : %d binary images with raw shape (%d,%d)." % (
        train_loader.dataset.data.shape[0],
        train_loader.dataset.data.shape[1],
        train_loader.dataset.data.shape[2]))
    print("Requested batch_size %d, so each epoch consists of %d updates" % (
        args.batch_size, len(train_loader)))

    ## Create an optimizer linked to the model parameters
    # Given gradients computed by pytorch, this optimizer handle update steps to params
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create batch of test data to visualize reconstructions for over training
    fixed_x_ND, _ = next(iter(test_loader))
    fixed_x_ND = fixed_x_ND.view(-1, model.n_dims_data)[:25]
    P = int(np.sqrt(model.n_dims_data))
    save_image(
        fixed_x_ND.view(25, 1, P, P), 
        '%s-recon_images-groundtruth.png' % (args.filename_prefix),
        nrow=5, padding=4)

    ## Training loop that repeats for each epoch:
    #  -- perform minibatch training updates (one epoch = one pass thru dataset)
    #  -- for latest model, compute performance metrics on training set
    #  -- for latest model, compute performance metrics on test set
    for epoch in range(args.n_epochs + 1):
        if epoch > 0:
            model.train_for_one_epoch_of_gradient_update_steps(
                optimizer, train_loader, epoch, args)

        ## Only save results for epochs 0,1,2,3,4 and 10,20,30,...
        if epoch > 4 and epoch % 10 != 0:
            continue

        print('==== evaluation after epoch %d' % (epoch))

        ## Compute nELBO loss (bce + kl)
        tr_loss, tr_msg = eval_model_on_data(
            model, 'train', train_eval_loader, args)
        if epoch == 0:
            print(tr_msg) # descriptive stats of tr data
        print('  epoch %3d  on train per-image nELBO loss %.3f ' % (epoch, tr_loss))

        te_loss, te_msg = eval_model_on_data(
            model, 'test', test_loader, args)
        if epoch == 0:
            print(te_msg) # descriptive stats of test data
        print('  epoch %3d  on test per-image nELBO loss %.3f ' % (epoch, te_loss))

        current_sigma = torch.exp(model.log_q_sigma).detach().item()
        ## Write perf metrics to CSV file (so we can easily plot later)
        row_df = pd.DataFrame([[
                epoch, tr_loss, te_loss, args.lr, args.q_sigma_init, current_sigma]],
            columns=[
                'epoch', 'tr_nelbo_loss', 'te_nelbo_loss', 'lr', 'q_sigma_init', 'q_sigma'])
        csv_str = row_df.to_csv(
            None,
            float_format='%.8f',
            index=False,
            header=False if epoch > 0 else True,
            )
        if epoch == 0:
            # At start, write to a clean file with mode 'w'
            with open('%s_perf_metrics.csv' % args.filename_prefix, 'w') as f:
                f.write(csv_str)
        else:
            # Append to existing file with mode 'a'
            with open('%s_perf_metrics.csv' % args.filename_prefix, 'a') as f:
                f.write(csv_str)

        with torch.no_grad():
            ## Make pretty plots of random samples in code space decoding into data space
            sample = model.sample_from_trained_model(25)
            save_image(
                sample.view(25, 1, P, P), 
                '%s-sampled_images-epoch=%03d.png' % (args.filename_prefix, epoch),
                nrow=5, padding=4)

            ## Make pretty plots of deterministic reconstructions of fixed test batch 
            recon_x_ND = torch.sigmoid(model.decode(model.encode(fixed_x_ND)))
            save_image(
                recon_x_ND.view(25, 1, P, P), 
                '%s-recon_images-epoch=%03d.png' % (args.filename_prefix, epoch),
                nrow=5, padding=4)

            ## If dim(z)=2, make pretty plots of encoded 2D space, colored by sample
            if args.latent_dim == 2:
                for n_per_category in [600, 6000]:
                    z_AC_by_cat = encode_data_by_digit_category(model, train_eval_loader, n_per_category=n_per_category)
                    fpath = '%s-encodings_viz-samples=%03d-epoch=%03d.png' % (args.filename_prefix, n_per_category, epoch)
                    xlims = (-float(3.), float(3.))
                    im_handle = plot_encoding_colored_by_digit_category(z_AC_by_cat, xlims=xlims)
                    plt.savefig(fpath, bbox_inches='tight', pad_inches=0)
                    plt.close(im_handle)

        print("====  done with eval at epoch %d" % epoch)

    # Save model at the end of training
    model_fpath = '%s-model.pytorch' % (args.filename_prefix)
    model.save_to_file(model_fpath)
