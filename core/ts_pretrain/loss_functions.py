import torch
from torch import nn
import numpy as np
import numpy
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """ Masked MSE Loss, from mvts
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()

        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)

        return self.mse_loss(masked_pred, masked_true)


class TripletLossVaryingLength(torch.nn.modules.loss._Loss):
    """
    https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries/blob/master/losses/triplet_loss.py

    Triplet loss for representations of time series where the training set
    features time series with unequal lengths.
    Takes as input a tensor as the chosen batch to compute the loss,
    a PyTorch module as the encoder, a 3D tensor (`B`, `C`, `L`) containing the
    training set, where `B` is the batch size, `C` is the number of channels
    and `L` is the maximum length of the time series (NaN values representing
    the end of a shorter time series), as well as a boolean which, if True,
    enables to save GPU memory by propagating gradients after each loss term,
    instead of doing it after computing the whole loss.
    The triplets are chosen in the following manner. First the sizes of
    positive and negative samples are randomly chosen in the range of lengths
    of time series in the dataset. The size of the anchor time series is
    randomly chosen with the same length upper bound but the length of the
    positive samples as lower bound. An anchor of this length is then chosen
    randomly in the given time series of the train set, and positive samples
    are randomly chosen among subseries of the anchor. Finally, negative
    samples of the chosen length are randomly chosen in random time series of
    the train set.
    @param compared_length Maximum length of randomly chosen time series. If
           None, this parameter is ignored.
    @param nb_random_samples Number of negative samples per batch example.
    @param negative_penalty Multiplicative coefficient for the negative sample
           loss.
    """

    def __init__(self, compared_length, nb_random_samples, negative_penalty, min_len=2):
        super(TripletLossVaryingLength, self).__init__()
        self.compared_length = compared_length
        if self.compared_length is None:
            self.compared_length = numpy.inf
        self.nb_random_samples = nb_random_samples
        self.negative_penalty = negative_penalty
        self.min_len = min_len

    def forward(self, batch, encoder, train, save_memory=False, is_seq_first=True):

        device = batch.device

        if isinstance(train, np.ndarray):
            train = torch.from_numpy(train)

        if is_seq_first:
            batch = batch.permute(0, 2, 1)
            train = train.permute(0, 2, 1)

        batch_size = batch.size(0)
        train_size = train.size(0)
        max_length = train.size(2)

        # For each batch element, we pick nb_random_samples possible random
        # time series in the training set (choice of batches from where the
        # negative examples will be sampled)
        samples = numpy.random.choice(
            train_size, size=(self.nb_random_samples, batch_size)
        )
        samples = torch.LongTensor(samples)

        # Computation of the lengths of the relevant time series
        with torch.no_grad():
            lengths_batch = max_length - torch.sum(
                torch.isnan(batch[:, 0]), 1
            ).data.cpu().numpy()
            lengths_samples = numpy.empty(
                (self.nb_random_samples, batch_size), dtype=int
            )
            for i in range(self.nb_random_samples):
                lengths_samples[i] = max_length - torch.sum(
                    torch.isnan(train[samples[i], 0]), 1
                ).data.cpu().numpy()

        # Choice of lengths of positive and negative samples
        lengths_pos = numpy.empty(batch_size, dtype=int)
        lengths_neg = numpy.empty(
            (self.nb_random_samples, batch_size), dtype=int
        )
        for j in range(batch_size):

            high = min(self.compared_length, lengths_batch[j]) + 1
            low = min(high - 1, self.min_len)

            lengths_pos[j] = numpy.random.randint(
                low, high=high
            )
            for i in range(self.nb_random_samples):
                high = min(self.compared_length, lengths_samples[i, j]) + 1
                low = min(high - 1, self.min_len)

                lengths_neg[i, j] = numpy.random.randint(
                    low,
                    high=high
                )

        # We choose for each batch example a random interval in the time
        # series, which is the 'anchor'
        random_length = numpy.array([numpy.random.randint(
            lengths_pos[j],
            high=min(self.compared_length, lengths_batch[j]) + 1
        ) for j in range(batch_size)])  # Length of anchors
        beginning_batches = numpy.array([numpy.random.randint(
            0, high=lengths_batch[j] - random_length[j] + 1
        ) for j in range(batch_size)])  # Start of anchors

        # The positive samples are chosen at random in the chosen anchors
        # Start of positive samples in the anchors
        beginning_samples_pos = numpy.array([numpy.random.randint(
            0, high=random_length[j] - lengths_pos[j] + 1
        ) for j in range(batch_size)])
        # Start of positive samples in the batch examples
        beginning_positive = beginning_batches + beginning_samples_pos
        # End of positive samples in the batch examples
        end_positive = beginning_positive + lengths_pos

        # We randomly choose nb_random_samples potential negative samples for
        # each batch example
        beginning_samples_neg = numpy.array([[numpy.random.randint(
            0, high=lengths_samples[i, j] - lengths_neg[i, j] + 1
        ) for j in range(batch_size)] for i in range(self.nb_random_samples)])

        representation = torch.cat([encoder(
            batch[
            j: j + 1, :,
            beginning_batches[j]: beginning_batches[j] + random_length[j]
            ], is_permute=True
        ) for j in range(batch_size)])  # Anchors representations

        positive_representation = torch.cat([encoder(
            batch[
            j: j + 1, :,
            end_positive[j] - lengths_pos[j]: end_positive[j]
            ], is_permute=True
        ) for j in range(batch_size)])  # Positive samples representations

        size_representation = representation.size(1)
        # Positive loss: -logsigmoid of dot product between anchor and positive
        # representations
        loss = -torch.mean(torch.nn.functional.logsigmoid(torch.bmm(
            representation.view(batch_size, 1, size_representation),
            positive_representation.view(batch_size, size_representation, 1)
        )))

        # If required, backward through the first computed term of the loss and
        # free from the graph everything related to the positive sample
        if save_memory:
            loss.backward(retain_graph=True)
            loss = 0
            del positive_representation
            torch.cuda.empty_cache()

        multiplicative_ratio = self.negative_penalty / self.nb_random_samples
        for i in range(self.nb_random_samples):
            # Negative loss: -logsigmoid of minus the dot product between
            # anchor and negative representations
            negative_representation = torch.cat([encoder(
                train[samples[i, j]: samples[i, j] + 1].to(device)[
                :, :,
                beginning_samples_neg[i, j]:
                beginning_samples_neg[i, j] + lengths_neg[i, j]
                ], is_permute=True
            ) for j in range(batch_size)])
            loss += multiplicative_ratio * -torch.mean(
                torch.nn.functional.logsigmoid(-torch.bmm(
                    representation.view(batch_size, 1, size_representation),
                    negative_representation.view(
                        batch_size, size_representation, 1
                    )
                ))
            )

            # If required, backward through the first computed term of the loss
            # and free from the graph everything related to the negative sample
            # Leaves the last backward pass to the training procedure
            if save_memory and i != self.nb_random_samples - 1:
                loss.backward(retain_graph=True)
                loss = 0
                del negative_representation
                torch.cuda.empty_cache()

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        """
        https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/

        Parameters
        ----------
        batch_size
        temperature
        """
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))

    @property
    def negatives_mask(self):
        return (~torch.eye(self.batch_size * 2, self.batch_size * 2, dtype=bool)).float()

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)

        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask.to(similarity_matrix.device) * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class TemporalContrastingLoss(nn.Module):
    def __init__(self,
                 k_ratio=0.2,
                 max_k_sample=16):
        super().__init__()
        self.k_ratio = k_ratio
        self.max_k_sample = max_k_sample

    def _neg_aug_for_batch_seq(self, batch_seq):
        neg_aug_seq = []
        for batch_i, _ in enumerate(batch_seq):
            neg_time_step_seq = torch.cat([batch_seq[:batch_i, :, :], batch_seq[batch_i + 1:, :, :]])
            neg_time_step_seq = torch.flatten(neg_time_step_seq,
                                              start_dim=0,
                                              end_dim=1)  # ((batch_size - 1) * time_step, feat_dim)
            neg_aug_seq.append(neg_time_step_seq)
        neg_aug_seq = torch.stack(neg_aug_seq)  # (batch_size, (batch_size - 1) * time_step, feat_dim)
        return neg_aug_seq

    def forward(self,
                encoder,
                weak_aug_seq,
                strong_aug_seq
                ):
        """

        Parameters
        ----------
        encoder
        weak_aug_seq: (batch_size, seq_len, feature_dim)
        strong_aug_seq

        Returns
        -------

        """
        if not isinstance(weak_aug_seq, torch.Tensor):
            weak_aug_seq = torch.Tensor(weak_aug_seq)
        if not isinstance(strong_aug_seq, torch.Tensor):
            strong_aug_seq = torch.Tensor(strong_aug_seq)

        max_seq_len = weak_aug_seq.shape[1]
        split_index = max_seq_len - int(max_seq_len * self.k_ratio)
        end_index = split_index + self.max_k_sample
        total = strong_aug_seq[:, split_index:end_index, :].shape[1]

        # before t context embedding (batch_size, project_dim)
        weak_context_embedding = encoder(weak_aug_seq[:, :split_index, :])
        strong_context_embedding = encoder(strong_aug_seq[:, :split_index, :])

        # after t timestep embedding (batch_size, time_step, project_dim)
        weak_future_seq = weak_aug_seq[:, split_index:end_index, :]
        strong_future_seq = strong_aug_seq[:, split_index:end_index, :]
        weak_future_t_embedding = encoder.input_project(weak_future_seq)
        strong_future_t_embedding = encoder.input_project(strong_future_seq)

        # do timestep neg sampling

        neg_weak_future_seq = self._neg_aug_for_batch_seq(weak_future_seq)
        neg_strong_future_seq = self._neg_aug_for_batch_seq(strong_future_seq)

        # neg_weak_feature_t_embedding: (batch_size, (batch_size - 1) * time_step, project_dim)
        neg_weak_feature_t_embedding = encoder.input_project(neg_weak_future_seq)
        neg_strong_feature_t_embedding = encoder.input_project(neg_strong_future_seq)

        # (batch_size, 1, project_dim), (batch_size, project_dim, time_step) -> (batch_size, time_step)
        pos_loss = torch.bmm(weak_context_embedding.unsqueeze(1), strong_future_t_embedding.permute(0, 2, 1)).squeeze(
            1) + torch.bmm(strong_context_embedding.unsqueeze(1), weak_future_t_embedding.permute(0, 2, 1)).squeeze(1)

        # (batch_size, 1, project_dim), (batch_size, project_dim, (batch_size - 1) * time_step) -> (batch_size, (batch_size - 1) * time_step)
        neg_loss = torch.bmm(weak_context_embedding.unsqueeze(1),
                             neg_strong_feature_t_embedding.permute(0, 2, 1)).squeeze(1) + \
                   torch.bmm(strong_context_embedding.unsqueeze(1),
                             neg_weak_feature_t_embedding.permute(0, 2, 1)).squeeze(1)
        loss = (-1 * (torch.sum(pos_loss) - torch.sum(neg_loss))) / total
        loss = torch.sigmoid(loss)
        return loss


class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self,
                 alpha: float = 0.5,
                 temporal_unit: int = 0):
        assert alpha <= 1.0
        super(HierarchicalContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.temporal_unit = temporal_unit

    def hierarchical_contrastive_loss(self, z1, z2, alpha=0.5, temporal_unit=0):
        loss = torch.tensor(0., device=z1.device)
        d = 0
        while z1.size(1) > 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            if d >= temporal_unit:
                if 1 - alpha != 0:
                    loss += (1 - alpha) * self.temporal_contrastive_loss(z1, z2)
            d += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        if z1.size(1) == 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            d += 1
        return loss / d

    def instance_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if B == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=0)  # 2B x T x C
        z = z.transpose(0, 1)  # T x 2B x C
        sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # T x 2B x (2B-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]

        logits = -F.log_softmax(logits, dim=-1)

        i = torch.arange(B, device=z1.device)

        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        return loss

    def temporal_contrastive_loss(self, z1, z2):
        B, T = z1.size(0), z1.size(1)
        if T == 1:
            return z1.new_tensor(0.)
        z = torch.cat([z1, z2], dim=1)  # B x 2T x C
        sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]  # B x 2T x (2T-1)
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        return loss

    def forward(self, z1, z2):
        return self.hierarchical_contrastive_loss(z1, z2, self.alpha, self.temporal_unit)
