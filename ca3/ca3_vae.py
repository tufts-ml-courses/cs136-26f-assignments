'''
Starter Code for VAE trained to minimize negative ELBO

TODO LIST
---------
* [ ] FIX draw_sample_from_q
* [ ] FIX calc_nelbo
* [ ] FIX sample_from_trained_model
'''
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(
            self,
            q_sigma_init=0.05,
            n_dims_latent=2,
            n_dims_data=400,
            activation_fn=F.relu):
        super(VariationalAutoencoder, self).__init__()
        self.n_dims_latent = n_dims_latent
        self.n_dims_data = n_dims_data
        self.activation_fn = activation_fn
        self.kwargs = dict(
            n_dims_latent=n_dims_latent, n_dims_data=n_dims_data, 
            activation_fn=activation_fn, q_sigma_init=q_sigma_init)
        
        self.log_q_sigma = nn.Parameter(torch.Tensor([float(q_sigma_init)]).log())

        self.encoder_layer_1 = nn.Linear(n_dims_data, 256)
        self.encoder_layer_2 = nn.Linear(256, 128)
        self.encoder_layer_3 = nn.Linear(128, n_dims_latent)

        self.decoder_layer_1 = nn.Linear(n_dims_latent, 128)
        self.decoder_layer_2 = nn.Linear(128, 256)
        self.decoder_layer_3 = nn.Linear(256, n_dims_data)


    def forward(self, x_ND):
        """ Encode then decode data to obtain variational posterior and reconstruction

        Args
        ----
        x_ND : tensor-like, N x D
            Batch of N images, each reshaped into a "flat" vector of length D.

        Returns
        -------
        recon_logit_ND : tensor-like, N x D
            logits of reconstruction probabilities, of the same shape as the input batch.

        mu_NC : tensor-like, N x C
            Mean of the encoding for each of the N images in minibatch.
        """
        mu_NC = self.encode(x_ND)
        z_NC = self.draw_sample_from_q(mu_NC)
        return self.decode(z_NC), mu_NC

    def encode(self, x_ND):
        h1_NH = self.activation_fn(self.encoder_layer_1(x_ND))
        h2_NH = self.activation_fn(self.encoder_layer_2(h1_NH))
        mu_NC = self.encoder_layer_3(h2_NH)
        return mu_NC

    def decode(self, z_NC):
        h1_NH = self.activation_fn(self.decoder_layer_1(z_NC))
        h2_NH = self.activation_fn(self.decoder_layer_2(h1_NH))
        recon_logit_ND = self.decoder_layer_3(h2_NH)
        return recon_logit_ND

    def draw_sample_from_q(self, mu_NC):
        ''' Draw sample from the probabilistic encoder q(z|mu(x), \\sigma)

        We assume that "q" is Normal with:
        * mean mu (argument of this function, outputted by the encoder)
        * stddev q_sigma (global attribute of this class, use self.log_q_sigma)

        Args
        ----
        mu_NC : tensor-like, N x C
            Mean of the encoding for each of the N images in minibatch.

        Returns
        -------
        z_NC : tensor-like, N x C
            Exactly one sample vector for each of the N images in minibatch.
        '''
        N = mu_NC.shape[0]
        C = self.n_dims_latent

        # exponentiate parameter to get positive stdev
        q_sigma = torch.exp(self.log_q_sigma)

        ## TODO
        # Using reparameterization trick,
        # Write a procedure here to return z_NC, a valid draw from q
        # Note that you are being handed a BATCH of N latent variables.
        # Your epsilon should be different for every variable in the batch.
        # You are not allowed to use torch.distributions; 
        # You must write the reparameterization trick *manually*
        return mu_NC

    def calc_nelbo(self, x_ND, do_rescale=True):
        ''' Calculates the negative ELBO for a batch 
        
        The reconstruction component is estimated with 1 Monte Carlo sample
        '''
        recon_logit_ND, mu_NC = self.forward(x_ND)

        # TODO: Add the KL divergence component of the loss, as outlined
        # in the assignment html
        kl_loss = 0.
        
        # TODO: replace this reconstruction MSE with -log p(x | z)
        # Make sure your result is a SUM over dimensions and the batch
        # (We turn it into a mean below based on if do_rescale is set)
        recon_probs = torch.sigmoid(recon_logit_ND)
        recon_loss = torch.sum((recon_probs - x_ND) ** 2)

        total_loss = kl_loss + recon_loss
        if do_rescale:
            N, D = x_ND.shape
            return total_loss / float(N)
        else:
            return total_loss

    def sample_from_trained_model(self, n_samples):
        ''' Draws samples from p(z) and computes the means/probs of p(x|z) for those samples

        If we wanted "true" samples from the generative model, we could also 
        sample from p(x|z). It often makes better visualizations to sample in
        the latent space but report the mean in the data space

        Returns
        -------
        x_probs_MD : tensor-like, M x D
            n_samples (M) draws from the generative model, 
            using the fixed prior and trained decoder.

        '''

        # this line "turns off" automatic differentiation.
        # At training time we don't need sample from p;
        # We only do this at evaluation time when we don't want grads.
        # Do all your computation within this no_grad environment 
        with torch.no_grad():

            # TODO fix me. Instead of returning random noise, 
            # draw from the generative model defined by the VAE
            x_MD = torch.randn((n_samples, self.n_dims_data))
            return torch.sigmoid(x_MD) # returns probabilities, not logits

    def train_for_one_epoch_of_gradient_update_steps(
            self, optimizer, train_loader, epoch, args):
        ''' Perform one epoch of gradient updates on provided model & data.

        Steps through dataset, one minibatch at a time.
        At each minibatch, we compute the gradient and step in that direction.

        Post Condition
        --------------
        This object's internal parameters are updated
        '''
        self.train() # mark as ready for training
        n_batch = len(train_loader)
        loss_per_batch = np.zeros(n_batch)
        num_batch_before_print = int(np.ceil(n_batch/5))

        for batch_idx, (batch_data, _) in enumerate(train_loader):
            # Reshape the data from 3dim to 2dim
            batch_x_ND = batch_data.view(-1, self.n_dims_data)
            
            # Zero out any stored gradients attached to the optimizer
            optimizer.zero_grad()

            # Compute the loss (and the required reconstruction as well)
            loss = self.calc_nelbo(batch_x_ND)

            # Track loss over all batches
            loss_per_batch[batch_idx] = loss.item()

            # Compute the gradient of the loss wrt model parameters
            # (gradients are stored as attributes of parameters of 'model')
            loss.backward()

            # Take an optimization step (gradient descent step)
            optimizer.step() # side-effect: updates internals of self's model!

            # Done with this batch. Write a progress update to stdout, move on.
            is_last = batch_idx + 1 == len(train_loader)
            if (batch_idx + 1) % num_batch_before_print  == 0 or is_last:
                print("  epoch %3d | frac_seen %.2f | avg loss % .4f | batch loss % .4f" % (
                    epoch, (1+batch_idx) / float(len(train_loader)),
                    loss_per_batch[:(1+batch_idx)].mean(),
                    loss.item()
                    ))

    def save_to_file(self, fpath):
        """ Save this model to file
        """
        state_dict = self.state_dict()
        state_dict['kwargs'] = self.kwargs
        torch.save(state_dict, fpath)

    @classmethod
    def save_model_to_file(cls, model, fpath):
        """ Save given model to file (class method)
        """
        model.save_to_file(fpath)

    @classmethod
    def load_model_from_file(cls, fpath):
        """ Load from file (class method)

        Usage
        -----
        >>> VariationalAutoencoder.load_model_from_file('path/to/model.pytorch')
        """
        state_dict = torch.load(fpath, weights_only=False)
        kwargs = state_dict.pop('kwargs')
        model = cls(**kwargs)
        assert 'kwargs' not in state_dict
        model.load_state_dict(state_dict)
        return model
