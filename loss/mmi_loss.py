"""MMI loss interface between pykaldi and pytorch"""
import logging
from itertools import repeat

import torch
from torch.autograd import Function
from .loss import Loss, add_loss

from ..utils.common import read_log_prior

logger = logging.getLogger(__name__)


def calcgrad(log_like, prob, trans_model, cur_ali, cur_lat, old_am_scale, am_scale, lm_scale, ismooth, drop_frame):
    """Calculate the loss and grad of one utterence"""
    # Lazy init pykaldi
    import kaldi.lat
    import kaldi.fstext
    import kaldi.fstext.utils
    from kaldi.fstext import LatticeVectorFst
    from kaldi.matrix import Matrix
    from kaldi.decoder import DecodableMatrixScaled

    if old_am_scale != 1.0:
        kaldi.fstext.utils.scale_lattice(kaldi.fstext.utils.acoustic_lattice_scale(old_am_scale), cur_lat)

    # !!! do not put it into function !!!
    # python will not copy it for you
    kaldi_m = Matrix(log_like)

    ds = DecodableMatrixScaled(kaldi_m, 1.0)
    kaldi.lat.functions.rescore_lattice(ds, cur_lat)

    if am_scale != 1.0 or lm_scale != 1.0:
        kaldi.fstext.utils.scale_lattice(kaldi.fstext.utils.lattice_scale(lm_scale, am_scale), cur_lat)
    total_like, arc_post, ac_like = kaldi.lat.functions.lattice_forward_backward(cur_lat)
    assert len(cur_ali) >= len(
        arc_post), f"length cur_ali small than length lat. len_ali = {len(cur_ali)}, len_lat = {len(arc_post)}"

    num_like = 0
    num_post = 0
    num_drop = 0
    grad = torch.zeros(log_like.shape)

    for t in range(len(arc_post)):
        posts = arc_post[t]
        for arc in posts:
            pdf = trans_model.transition_id_to_pdf(arc[0]) if trans_model is not None else arc[0] - 1
            grad[t][pdf] += arc[1]
        pdf = trans_model.transition_id_to_pdf(cur_ali[t]) if trans_model is not None else cur_ali[t] - 1
        num_like += log_like[t][pdf]
        num_post += grad[t][pdf]

        if grad[t][pdf] < 1e-20 and drop_frame:
            num_drop += 1
            grad[t] = 0
        else:
            grad[t][pdf] -= 1
            if ismooth > 0:
                prob[t][pdf] -= 1
                grad[t] = grad[t] * (1 - ismooth) + prob[t] * ismooth * 0.1
    cost = num_like * am_scale - total_like
    logger.debug("MMI value: ", cost / len(arc_post))
    return (cost, grad, num_like, num_post, num_drop, len(arc_post))


class _MMI(Function):

    @staticmethod
    def forward(ctx, act, ali, lattice, trans_model, old_am_scale,
                am_scale, lm_scale, ismooth, drop_frame,
                log_prior=None, reduction='sum'):
        num_utts = act.size(0)
        log_like = act.cpu().detach().numpy()
        if log_prior is not None:
            log_like -= log_prior
        if ismooth > 0:
            prob = act.softmax(dim=-1).cpu()
        else:
            prob = [None] * num_utts

        arg_list = [log_like, prob, repeat(trans_model), ali, lattice, repeat(old_am_scale), repeat(am_scale),
                    repeat(lm_scale), repeat(ismooth), repeat(drop_frame)]

        rets = map(calcgrad, *arg_list)

        costs_list, grads, num_likes, num_posts, num_drops, num_frames = zip(*rets)
        costs = torch.Tensor(costs_list)
        grads = torch.stack(grads).to(act.device)
        assert grads.size() == act.size()

        num_like = sum(num_likes)
        num_post = sum(num_posts)
        num_drop = sum(num_drops)
        num_frames = sum(num_frames)

        if reduction == 'sum':
            loss = costs.sum()
            ctx.grads = grads
        elif reduction == 'mean':
            loss = costs.mean()
            ctx.grads = grads / costs.numel()
        elif reduction == 'none':
            raise NotImplementedError

        statistic = [num_like, num_post, num_frames, num_drop]
        statistic_ret = {
            'num_like': num_like,
            'num_post': num_post,
            'total_frames': num_frames,
            'dropped_frames': num_drop,
        }
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        out_none = [None] * 9
        grad_in = tuple([grad_output * ctx.grads, *out_none])
        return grad_in


mmi = _MMI.apply


@add_loss('mmi')
class MMILoss(Loss):
    """MMI loss criterion

    It reads the lattices and drop the ones not in ``batch['uid']``.

    ..warning:: It only works with sequence collector

    Parameters:
        transition_model (): usually None
        drop_frame (bool): whether to drop frame with low gradient (1e-20)
        old_am_scale (float): (default: `0`)
        am_scale (float): acoustic model scale
        lm_scale (float): language model scale
        ismooth (float): ismooth
        prior (Path): path to the log prior file (label.counts)

    Inputs:
        output: Tensor of (batch x seqLength x outputDim) containing output from network
        data_batch (dict): The data_batch dict from batch loader
    """

    def __init__(self, transition_model=None, drop_frame=True, old_am_scale=0,
                 am_scale=1, lm_scale=1, ismooth=0, prior=None):
        super().__init__()
        self.transition_model = transition_model
        self.drop_frame = drop_frame
        self.old_am_scale = old_am_scale
        self.am_scale = am_scale
        self.lm_scale = lm_scale
        self.ismooth = ismooth
        if prior is not None:
            self.prior = read_log_prior(prior)
        else:
            self.prior = None

    def extra_repr(self):
        line = f'drop_frame={self.drop_frame}\n'
        line += f'old_am_scale={self.old_am_scale}\n'
        line += f'am_scale={self.am_scale}\n'
        line += f'lm_scale={self.lm_scale}\n'
        line += f'ismooth={self.ismooth}'
        return line

    def forward(self, output, data_batch):
        lattice = self._read_lat(data_batch['uid'], data_batch['extra']['lattices'])[0]
        ali = data_batch['label'].tensor
        output = output.tensor
        loss = mmi(output, ali, lattice, self.transition_model,
                   self.old_am_scale, self.am_scale, self.lm_scale,
                   self.ismooth, self.drop_frame, self.prior)

        statistic_ret = {
            'loss': loss.item(),
            # 'num_like': num_like,
            # 'num_post': num_post,
            'total_frames': data_batch['label'].length.sum().item(),
            # 'dropped_frames': num_drop,
        }

        return loss, statistic_ret

    def log_line(self, reduced_stat):
        loss_per_frame = reduced_stat['loss'] / reduced_stat['total_frames']
        # dropped_frames = reduced_stat['dropped_frames']
        return f'Loss {loss_per_frame:.2f}, dropped frames {0.5:.0f}'

    def _read_lat(self, uid, lattices):
        """A hacking method to read lattice in main process

        Args:
            uid (tuple[str]): an ordered tuple of utterance ids in current batch
            lattices (list[lat_rspec]): a list of lattice rspecs, a flag may
                                        be used as an indicator of restarting

        Returns:
            list[list[Lattice]]: the list of list of Kaldi lattice objects
        """
        # Lazy init
        # Workaround of lattice.from_bytes memory leaking
        from kaldi.util.table import SequentialLatticeReader

        if lattices[0][0] is not None:
            self.lattice_readers = []
            for lat_rspecs in lattices:
                self.lattice_readers.append(iter(SequentialLatticeReader(lat_rspecs[0])))

        lat_lists = []
        for lat_reader, lat_rspecs in zip(self.lattice_readers, lattices):
            lat_list = []
            for cur_id, _ in zip(uid, lat_rspecs):
                lat_id, lattice = next(lat_reader)
                while lat_id != cur_id:
                    logger.warning(f'Drop lattice of {lat_id}')
                    lat_id, lattice = next(lat_reader)
                lat_list.append(lattice)
            lat_lists.append(lat_list)
        return lat_lists
