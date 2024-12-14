import torch
import torch.nn as nn

from strl.modules.recurrent_modules import RecurrentPredictor
from strl.utils.general_utils import batch_apply, ParamDict, AttrDict
from strl.utils.pytorch_utils import get_constant_parameter, ResizeSpatial, RemoveSpatial
from strl.models.skill_prior_mdl import SkillPriorMdl, ImageSkillPriorMdl
from strl.modules.subnetworks import Predictor, VQPredictor, BaseProcessingLSTM, Encoder, VQCDTPredictor
from strl.modules.variational_inference import MultivariateGaussian
from strl.components.checkpointer import load_by_key, freeze_modules
from strl.modules.vq_vae import VQEmbedding
from strl.modules.Categorical import Categorical


class SkillPriorVQMdl(SkillPriorMdl):
    """SPiRL model with VQ low-level skill decoder."""

    def build_network(self):
        assert not self._hp.use_convs  # currently only supports non-image inputs
        self.q = self._build_inference_net()  # encoder
        # self.decoder = VQPredictor(self._hp,
        #                            input_size=self._hp.nz_vae,
        #                            output_size=self._hp.action_dim,
        #                            mid_size=self._hp.nz_mid_prior)
        self.decoder = RecurrentPredictor(self._hp,
                                          input_size=self._hp.action_dim + self._hp.nz_vae,
                                          output_size=self._hp.action_dim)
        self.decoder_input_initalizer = self._build_decoder_initializer(size=self._hp.action_dim)
        self.decoder_hidden_initalizer = self._build_decoder_initializer(size=self.decoder.cell.get_state_size())

        self.p = self._build_prior_ensemble()
        self.codebook = self._build_codebook()
        self.log_sigma = get_constant_parameter(0., learnable=False)
        self.load_weights_or_freeze()

    def forward(self, inputs, use_learned_prior=False):
        """Forward pass of the VQ SPIRL model.
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        :arg use_learned_prior: if True, decodes samples from learned prior instead of posterior, used for RL
        """
        output = AttrDict()
        inputs.observations = inputs.actions  # for seamless evaluation

        # encode
        output.z_e_x = self.encode(inputs)

        output.z_q_x_st, output.z_q_x, output.indices = self.codebook.straight_through(output.z_e_x)

        # decode
        assert self._regression_targets(inputs).shape[1] == self._hp.n_rollout_steps
        output.reconstruction = self.decode(output.z_q_x_st,
                                            cond_inputs=self._learned_prior_input(inputs),
                                            steps=self._hp.n_rollout_steps)

        # infer learned skill prior
        output.q_hat = self.compute_learned_prior(self._learned_prior_input(inputs))
        output.prior_entropy = output.q_hat.entropy().mean()

        return output

    def loss(self, model_output, inputs):
        """Loss computation of the VQ SPIRL model.
        :arg model_output: output of SPIRL model forward pass
        :arg inputs: dict with 'states', 'actions', 'images' keys from data loader
        """
        losses = AttrDict()

        mse_loss = torch.nn.MSELoss()
        ce_loss = torch.nn.CrossEntropyLoss()

        # reconstruction loss, assume unit variance model output Gaussian
        losses.rec_mse = mse_loss(model_output.reconstruction, inputs.actions)

        # VQ loss
        losses.vq_loss = mse_loss(model_output.z_q_x, model_output.z_e_x.detach())

        # commitment loss
        losses.commitment_loss = self._hp.commitment_beta * mse_loss(model_output.z_e_x,
                                                                     model_output.z_q_x.detach())

        # learned skill prior net loss
        # losses.prior_loss = loss_cret(model_output.q_hat, model_output.z_q_x)
        losses.prior_loss = ce_loss(model_output.q_hat.prob.probs, model_output.indices)

        losses.total = losses.rec_mse + losses.vq_loss + losses.commitment_loss + losses.prior_loss
        return losses

    def encode(self, inputs):
        # run inference with state sequence conditioning
        inf_input = torch.cat((inputs.actions, self._get_seq_enc(inputs)), dim=-1)
        return self.q(inf_input)[:, -1]

    def _build_inference_net(self):
        # condition inference on states since decoder is conditioned on states too
        input_size = self._hp.action_dim + self.prior_input_size
        return torch.nn.Sequential(
            BaseProcessingLSTM(self._hp, in_dim=input_size, out_dim=self._hp.nz_enc),
            torch.nn.Linear(self._hp.nz_enc, self._hp.nz_vae)
        )

    def _build_prior_net(self):
        return VQPredictor(self._hp, input_size=self.prior_input_size, output_size=self._hp.codebook_K,
                           num_layers=self._hp.num_prior_net_layers, mid_size=self._hp.nz_mid_prior)

    def _compute_learned_prior(self, prior_mdl, inputs):
        return Categorical(logits=prior_mdl(inputs), codebook=self.codebook)

    def _build_codebook(self):
        return VQEmbedding(self._hp.codebook_K, self._hp.nz_vae)

    def _get_seq_enc(self, inputs):
        return inputs.states[:, :-1]

    def enc_obs(self, obs):
        """Optionally encode observation for decoder."""
        return obs

    def load_weights_and_freeze(self):
        """Optionally loads weights for components of the architecture + freezes these components."""
        if self._hp.embedding_checkpoint is not None:
            print("Loading pre-trained embedding from {}!".format(self._hp.embedding_checkpoint))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'decoder', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'q', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.embedding_checkpoint, 'codebook', self.state_dict(), self.device))
            freeze_modules([self.decoder, self.q, self.codebook])
        else:
            super().load_weights_and_freeze()

    def load_weights_or_freeze(self):
        if hasattr(self._hp, 'cdt_embedding_checkpoint') and self._hp.cdt_embedding_checkpoint is not None:
            print("Loading pre-trained embedding from {}!".format(self._hp.cdt_embedding_checkpoint))
            self.load_state_dict(
                load_by_key(self._hp.cdt_embedding_checkpoint, 'decoder', self.state_dict(), self.device))
            self.load_state_dict(load_by_key(self._hp.cdt_embedding_checkpoint, 'q', self.state_dict(), self.device))
            self.load_state_dict(
                load_by_key(self._hp.cdt_embedding_checkpoint, 'codebook', self.state_dict(), self.device))
            if self._hp.if_freeze:
                freeze_modules([self.decoder, self.q, self.codebook])
                print('freeze!')

    def _log_losses(self, losses, step, log_images, phase):
        for name, loss in losses.items():
            self._logger.log_scalar(loss, name + '_loss', step, phase)
            # if 'breakdown' in loss and log_images:
            #     self._logger.log_graph(loss.breakdown, name + '_breakdown', step, phase)

    @property
    def enc_size(self):
        return self._hp.state_dim
        # return 30
