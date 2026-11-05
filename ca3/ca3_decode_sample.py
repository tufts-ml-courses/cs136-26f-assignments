import argparse
from torchvision.utils import save_image
import torch
import numpy as np
from ca3_vae import VariationalAutoencoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Decode specific 2d latent to an MNIST image')

    parser.add_argument(
        '--fpath', type=str,
        help='file path to trained model')
    parser.add_argument(
        '--z1', type=float, default=0.,
        help='1st dimension to decode')
    parser.add_argument(
        '--z2', type=float, default=0.,
        help='2nd dimension to decode')
    args = parser.parse_args()

    model = VariationalAutoencoder.load_model_from_file(args.fpath)
    model.eval()
    with torch.no_grad():
        x_D = torch.sigmoid(model.decode(torch.tensor([[args.z1, args.z2]])))

    P = int(np.sqrt(model.n_dims_data))
    save_image(
        x_D.view(1, 1, P, P), 
        'gen_images-z1-%.3f-z2-%.3f.png' % (args.z1, args.z2),
        nrow=1, padding=4)