#This code is based on: https://github.com/SimonKohl/probabilistic_unet
import math
import torch
from torch import nn
from Probabilistic_Unet_Pytorch.unet_blocks import *
from Probabilistic_Unet_Pytorch.unet import Unet
from Probabilistic_Unet_Pytorch.utils import init_weights,init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from pionono_models.model_headless import UnetHeadless
from lib.adaptive_sine_lora import AdaSineLoRALinear
from lib.sigreg import SIGReg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Conv_block(nn.Module):
    def __init__(self, input_channels, num_filters, padding=True):
        super(Conv_block, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters

        layers = []
        for i in range(len(self.num_filters)):

            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(0.1))
            layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class Projection(nn.Module):
    def __init__(
        self,
        num_experts,
        latent_dim,
        harmonizer_hidden=64,
        omega0=300.0,
        lora_rank=4,
        rater_embed_dim=8,
    ):
        super(Projection, self).__init__()
        self.num_experts = num_experts
        self.latent_dim = latent_dim
        self.shared_head = Conv_block(16, [32, 16, latent_dim], 3)
        self.rater_embedding = nn.Embedding(num_experts, rater_embed_dim)
        cond_dim = latent_dim + rater_embed_dim
        self.cond_mlp = nn.Sequential(
            nn.LayerNorm(cond_dim),
            nn.Linear(cond_dim, harmonizer_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(harmonizer_hidden, latent_dim),
        )
        rank = min(lora_rank, latent_dim)
        self.query_proj = AdaSineLoRALinear(
            latent_dim,
            latent_dim,
            r=rank,
            cond_dim=latent_dim,
            harmonizer_hidden=harmonizer_hidden,
            omega0=omega0,
            init_identity=True,
        )
        self.key_proj = nn.Linear(latent_dim, latent_dim)
        with torch.no_grad():
            self.key_proj.weight.zero_()
            self.key_proj.weight.copy_(
                torch.eye(latent_dim, dtype=self.key_proj.weight.dtype, device=self.key_proj.weight.device)
            )
            if self.key_proj.bias is not None:
                self.key_proj.bias.zero_()
        # self.out_proj = AdaSineLoRALinear(
        #     latent_dim,
        #     latent_dim,
        #     r=rank,
        #     cond_dim=latent_dim,
        #     harmonizer_hidden=harmonizer_hidden,
        #     omega0=omega0,
        #     init_identity=True,
        # )
        self.pooling_layer = nn.AdaptiveAvgPool2d([1,1])
        self.activation = torch.nn.Softmax(dim=2)

    def _build_cond(self, z_set, idx):
        bs = z_set.shape[0]
        z_summary = z_set.mean(dim=1)
        rater_ids = torch.full((bs,), idx, dtype=torch.long, device=z_set.device)
        rater_feat = self.rater_embedding(rater_ids)
        cond = torch.cat([z_summary, rater_feat], dim=-1)
        return self.cond_mlp(cond)

    def forward(self, feature_map, z_set, idx):
        feats = self.shared_head(feature_map)
        bs = feature_map.shape[0]
        global_z = self.pooling_layer(feats).view(bs, self.latent_dim, -1).permute(0,2,1)
        cond = self._build_cond(z_set, idx)
        q = self.query_proj(global_z, cond=cond)
        k = self.key_proj(z_set)
        similarity = torch.bmm(q, k.permute(0,2,1)) / math.sqrt(self.latent_dim)
        similarity = self.activation(similarity)
        output = torch.bmm(similarity, z_set)
        #output = self.out_proj(output, cond=cond)
        return output

class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            #To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += 4

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            
            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
            
            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_per_block-1):
                layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output

class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """
    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
        else:
            self.name = 'Prior'
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block, initializers, posterior=self.posterior)
        self.conv_layer = nn.Conv2d(num_filters[-1], 2 * self.latent_dim, (1,1), stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        #If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            self.show_img = input
            self.show_seg = segm
            # segm = torch.unsqueeze(segm, 1)
            input = torch.cat((input, segm), dim=1)
            self.show_concat = input
            self.sum_input = torch.sum(input)

        encoding = self.encoder(input)
        self.show_enc = encoding

        #We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        #Convert encoding to 2 x latent dim and split up for mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)

        #We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:,:self.latent_dim]
        log_sigma = mu_log_sigma[:,self.latent_dim:]

        #This is a multivariate normal with diagonal covariance matrix sigma
        #https://github.com/pytorch/pytorch/pull/11178
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)),1)
        return dist

class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """
    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers, use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels #output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2,3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb 
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            #Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(16+self.latent_dim, self.num_filters[0], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters[0], self.num_filters[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)
        self.activation = torch.nn.Softmax(dim=1)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z, use_softmax=True):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z,2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z,3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            #Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            output = self.last_layer(output)
            if use_softmax:
                output = self.activation(output)
            return output


class PIRA(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, num_filters=[16,32,64,128,256],
                 latent_dim=6, no_convs_fcomb=4, num_experts=4, reg_factor=1.0, original_backbone=False):
        super(PIRA, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = 3
        self.no_convs_fcomb = no_convs_fcomb
        self.num_experts = num_experts
        self.initializers = {'w':'he_normal', 'b':'normal'}
        self.reg_factor = reg_factor
        self.original_backbone = original_backbone
        if self.original_backbone:
            self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, self.initializers, apply_last_layer=False, padding=True).to(device)
        else:
            self.unet = UnetHeadless(input_channels).to(device)

        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim,  self.initializers,).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block, self.latent_dim, self.initializers, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes, self.no_convs_fcomb, {'w':'orthogonal', 'b':'normal'}, use_tile=True).to(device)
    
        self.proj_heads = Projection(self.num_experts, self.latent_dim).to(device)
        self.sigreg = SIGReg(num_slices=64, standardize=True).to(device)

    def forward(self, patch, segm=None, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, segm)
        self.prior_latent_space = self.prior.forward(patch)
        if self.original_backbone:
            self.unet_features = self.unet.forward(patch, False)
        else:
            self.unet_features = self.unet.forward(patch)

    def minmax(self, post_masks):
        gt_max_union = torch.max(post_masks, dim=1, keepdim=True).values
        gt_min_union = torch.min(post_masks, dim=1, keepdim=True).values
        masks = torch.cat([gt_max_union, gt_min_union],dim=1)
        return masks

    def get_z_set(self, dist, sample_num=400):
        z_set = []
        for _ in range(sample_num):
            z_case = dist.sample()
            z_set.append(z_case.unsqueeze(1))
        z_set = torch.cat(z_set, dim=1)
        return z_set

    def z_mapping(self, z_set):
        num_z = z_set.size(1)
        cases = [] #
        for idx in range(num_z):
            z_case = z_set[:,idx]
            case = self.fcomb.forward(self.unet_features, z_case, use_softmax=False)
            cases.append(case)
        samples = torch.cat(cases, dim=1)
        sample = torch.mean(samples, dim=1).unsqueeze(1)
        return sample

    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            #Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

    def prior_sampling(self, sample_num=20, training=False):
        samples = []
        for _ in range(sample_num):
            if training:
                z_prior = self.prior_latent_space.rsample()
            else:
                z_prior = self.prior_latent_space.sample()
            sample = self.fcomb.forward(self.unet_features, z_prior, use_softmax=False)
            samples.append(sample)
        return samples

    def posterior_sampling(self, sample_num=20, training=False):
        samples = []
        for _ in range(sample_num):
            if training:
                z_posterior = self.posterior_latent_space.rsample()
            else:
                z_posterior = self.posterior_latent_space.sample()
            sample = self.fcomb.forward(self.unet_features, z_posterior, use_softmax=False)
            samples.append(sample)
        return samples
    
    
    
    
    

    
    
    



    
    
    #     added for GED loss test
    
    
    
    
    def _logits_to_probs(self, logits_concat, num_classes):
        """
        logits_concat: [B, K*C, H, W]
        returns probs: [B, K, C, H, W]
        """
        B, KC, H, W = logits_concat.shape
        assert KC % num_classes == 0, "Channel dim must be K*C"
        K = KC // num_classes
        x = logits_concat.view(B, K, num_classes, H, W)
        if num_classes == 1:
            # binary: sigmoid over the single channel
            probs = torch.sigmoid(x)
        else:
            # multi-class: softmax over C (dim=2)
            probs = torch.softmax(x, dim=2)
        return probs
    
    # def _soft_dice_distance(self, p, y, eps=1e-6):
    #     """
    #     p: [B, C, H, W] probabilities
    #     y: [B, C, H, W] binary (0/1)
    #     returns: [B] distance per batch item
    #     """
    #     inter = (p * y).sum(dim=(1,2,3))
    #     denom = p.sum(dim=(1,2,3)) + y.sum(dim=(1,2,3))
    #     dice = (2.0 * inter + eps) / (denom + eps)
    #     return 1.0 - dice
    
    # def _mean_pairwise_distance(self, samples, dist_fn):
    #     """
    #     samples: [B, K, C, H, W] probabilities
    #     dist_fn: function([B,C,H,W],[B,C,H,W]) -> [B]
    #     returns: [B] mean over all unordered pairs
    #     """
    #     B, K, C, H, W = samples.shape
    #     if K < 2:
    #         return torch.zeros(B, device=samples.device, dtype=samples.dtype)
    #     # Compute average over all k<k'
    #     acc = 0.0
    #     count = 0
    #     for i in range(K):
    #         for j in range(i+1, K):
    #             acc = acc + dist_fn(samples[:, i], samples[:, j])
    #             count += 1
    #     return acc / max(count, 1)
    
    # def _ged_from_prior_samples(self, logits_concat, y, num_classes=1):
    #     """
    #     logits_concat: [B, K*C, H, W] from self.re_prior (logits)
    #     y: ground-truth mask -> [B, C, H, W] (binary)
    #     returns: scalar GED (mean over batch)
    #     """
    #     # 1) logits -> probs and split into K samples
    #     probs = self._logits_to_probs(logits_concat, num_classes)  # [B,K,C,H,W]
    #     B, K, C, H, W = probs.shape
    
    #     # 2) E[d(X,Y)]
    #     #    broadcast y to compare with each sample
    #     d_xy = []
    #     for k in range(K):
    #         d_xy.append(self._soft_dice_distance(probs[:, k], y))
    #     EdXY = torch.stack(d_xy, dim=1).mean(dim=1)  # [B]
    
    #     # 3) E[d(X,X')]
    #     EdXXp = self._mean_pairwise_distance(probs, self._soft_dice_distance)  # [B]
    
    #     ged = 2.0 * EdXY - EdXXp  # [B]
    #     return ged.mean()  # scalar
    
    
    
    

    
    
    
    
    def _ged_from_prior_samples_multi_rater(self, logits_concat, y_all, num_classes=1, eps=1e-6):
        """
        logits_concat: [B, K*C, H, W]  (prior samples, logits)
        y_all:         [B, M, H, W]    for binary
                        or [B, M, C, H, W] for multi-class one-hot
        returns: scalar GED averaged over batch
        """
        # 1) logits -> probs: [B,K,C,H,W]
        probs = self._logits_to_probs(logits_concat, num_classes)  # [B,K,C,H,W]
        B, K, C, H, W = probs.shape
    
        # 2) Ensure y has shape [B,M,C,H,W]
        if y_all.dim() == 4:           # binary [B,M,H,W] -> add C=1
            y_all = y_all.unsqueeze(2) # [B,M,1,H,W]
        y_all = y_all.float()
    
        # 3) E[d(X,Y)] over K (samples) and M (raters), using soft Dice distance
        P = probs.unsqueeze(1)          # [B,1,K,C,H,W]
        Y = y_all.unsqueeze(2)          # [B,M,1,C,H,W]
    
        inter = (P * Y).sum(dim=(3,4,5))                # [B,M,K]
        denom = P.sum(dim=(3,4,5)) + Y.sum(dim=(3,4,5)) # [B,M,K]
        dice  = (2.0 * inter + eps) / (denom + eps)     # [B,M,K]
        d_xy  = 1.0 - dice                              
        EdXY  = d_xy.mean(dim=(1,2))                    # [B]
    
        # 4) E[d(X,X')] across K samples (k<k')
        P1 = probs.unsqueeze(2)  # [B,K,1,C,H,W]
        P2 = probs.unsqueeze(1)  # [B,1,K,C,H,W]
        inter_xx  = (P1 * P2).sum(dim=(3,4,5))                 # [B,K,K]
        denom_xx  = P1.sum(dim=(3,4,5)) + P2.sum(dim=(3,4,5))  # [B,K,K]
        dice_xx   = (2.0 * inter_xx + eps) / (denom_xx + eps)  # [B,K,K]
        d_xx      = 1.0 - dice_xx
    
        # mask upper triangle (k<k')
        Kdim = d_xx.size(1)
        triu = torch.triu(torch.ones(Kdim, Kdim, device=d_xx.device, dtype=torch.bool), diagonal=1)
        # average over pairs
        EdXXp = d_xx[:, triu].view(B, -1).mean(dim=1)          # [B]
    
        ged = 2.0 * EdXY - EdXXp                                # [B]
        return ged.mean()

        
    
    
    
    
    
    
    
    

    def elbo(self, args, segm, criterion, analytic_kl=True):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
  
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calculate_posterior=False))

        #Here we use the posterior sample sampled above
        self.re_post = self.posterior_sampling(sample_num=1, training=True)[0]

        #Here we use the prior sample sampled above
        self.re_prior = torch.cat(self.prior_sampling(sample_num=args.prior_sample_num, training=True), dim=1)

        prior_bound = self.minmax(segm)
        prior_pred_max = torch.max(self.re_prior, dim=1, keepdim=True).values
        prior_pred_min = torch.min(self.re_prior, dim=1, keepdim=True).values
        pred_bound = torch.cat([prior_pred_max, prior_pred_min],dim=1)
        bound_loss = criterion(pred_bound, prior_bound)

        random_idx = np.random.randint(0, args.mask_num)
        random_label = segm[:,random_idx].unsqueeze(1)

        reconstruction_loss = criterion(self.re_post, random_label)
        self.reconstruction_loss = torch.sum(reconstruction_loss)
        self.bound_loss = torch.sum(bound_loss)
        
        
        
        
        
        # ======== NEW: GED term (uses prior samples vs. deterministic GT) ========
        # pick a single GT: if segm has shape [B,1,H,W], this just selects index 0
        # gt_one = segm[:, 0].unsqueeze(1)  # [B,1,H,W]; change if multi-class one-hot
        # num_classes = self.num_classes  # typically 1 here
        # ged_term = self._ged_from_prior_samples(self.re_prior, gt_one, num_classes=num_classes)
        ged_term = self._ged_from_prior_samples_multi_rater(self.re_prior, segm, num_classes=self.num_classes)


        # weight of GED in total objective (set in args)
        # ged_weight = getattr(args, "ged_lambda", 1.0)
        ged_weight = 0.25

        # SIGReg: push the *aggregate* posterior distribution toward N(0, I).
        # Draw 5 posterior samples per batch item so the EP test has enough
        # points (5 × B = 60 for B=12) to be meaningful with latent_dim=6.
        # KL handles per-sample posterior→prior alignment; SIGReg handles
        # the global latent geometry (prevents drift, collapse, anisotropy).
        sigreg_lambda = getattr(args, 'sigreg_lambda', 0.0)
        if sigreg_lambda > 0.0:
            z_post_aug = torch.cat(
                [self.posterior_latent_space.rsample() for _ in range(5)], dim=0
            )  # [5B, latent_dim]
            sigreg_loss = self.sigreg(z_post_aug)
        else:
            sigreg_loss = 0.0

        return -(self.reconstruction_loss + self.kl + args.beta * self.bound_loss
                 + ged_weight * ged_term + sigreg_lambda * sigreg_loss)

        

    def combined_loss(self, args, labels, loss_fct):
        elbo = self.elbo(args, labels, criterion=loss_fct)
        self.reg_loss = (l2_regularisation(self.posterior) + l2_regularisation(self.prior) + l2_regularisation(self.fcomb.layers)) * self.reg_factor
        loss = -elbo + self.reg_loss
        return loss

    def train_step(self, args, images, masks, loss_fct, stage = 1):
        if stage == 1:
            self.forward(images, masks, training=True)
            loss = self.combined_loss(args, masks, loss_fct)
            y_preds = self.re_post
        elif stage == 2:
            # Inline test_step so we can intercept per-expert z projections
            # for SIGReg without changing test_step's public signature.
            self.forward(images, None, training=False)
            z_set = self.get_z_set(self.prior_latent_space, sample_num=100)

            zs_expert, samples_expert = [], []
            for idx in range(self.num_experts):
                z_expert = self.proj_heads(self.unet_features, z_set, idx)  # [B, 1, D]
                zs_expert.append(z_expert.squeeze(1))                        # [B, D]
                samples_expert.append(self.z_mapping(z_expert))
            y_preds = torch.cat(samples_expert, dim=1)                       # [B, E, H, W]

            seg_loss = loss_fct(y_preds, masks)

            # SIGReg on combined expert latents [B*E, D]: prevents any expert
            # head from collapsing its projected representations.
            sigreg_lambda = getattr(args, 'sigreg_lambda', 0.0)
            if sigreg_lambda > 0.0:
                all_z = torch.cat(zs_expert, dim=0)   # [B*E, D]
                loss = seg_loss + sigreg_lambda * self.sigreg(all_z)
            else:
                loss = seg_loss
        return loss, y_preds

    def val_step(self, images, sample_num = 50):
        self.forward(images, None, training=False)
        y_pred = torch.cat(self.prior_sampling(sample_num=sample_num, training=False), dim=1)
        return y_pred

    def test_step(self, images, sample_num = 100):
        self.forward(images, None, training=False)
        z_set = self.get_z_set(self.prior_latent_space, sample_num=sample_num)
        zs_expert, samples_expert = [], []
        for idx in range(self.num_experts):
            z_expert = self.proj_heads(self.unet_features, z_set, idx)
            sample_expert = self.z_mapping(z_expert)
            zs_expert.append(z_expert)
            samples_expert.append(sample_expert)
        y_preds = torch.cat(samples_expert, dim=1)
        return y_preds
