import torch
import torch.nn as nn

from .loss import Loss

class logMSELoss(Loss):
    """A warpper of original pytorch CELoss

    It simply unpacks the output and batch into specific format, and computes
    some useful metrics for logging purposes.

    Args:
        output (tensor, B,T,C)
        data_batch (dict)
    """

    def __init__(self, reduction='sum'):
        super().__init__()
        self.loss_kernel = nn.MSELoss(reduction=reduction)

    def forward(self, output, data_batch):
        if 'skipped_label' in data_batch:
            label_field = data_batch['skipped_label']
        else:
            label_field = data_batch['label']
        if 'skipped_feat' in data_batch:
            feat_field = data_batch['skipped_feat']
        else:
            feat_field = data_batch['feat']
        label = label_field.tensor
        #feat = feat_field.tensor
        
        label_len = label_field.length
        output_len = output.length
        # one output for one label
        if not torch.all(label_len == output_len):
            raise RuntimeError(f'Length mismatch: label_len={label_len} \n !== output_len={output_len}')

        output = output.tensor
        '''
        if output.size() != label.size():
            raise RuntimeError(f'The model output size ({output.size()[:2]}) must '
                               f'match label size ({label.size()}) for first 2 dimensions. '
                               'Please check if you transposed the output correctly.')
        '''
        
        output = output.reshape(-1, output.shape[-1]).float()
        #output = output[:,1:64]
        output = torch.log10(torch.clamp(output,min=0.05))
        #print(output.size())
        #print(output[1:100:500,1:4:64])
        #label = label.cuda().view(-1)
        label = label.cuda().reshape(-1,label.shape[-1])
        #feat = feat.cuda().reshape(-1,feat.shape[-1]) 
        #noisy = feat.clone()
        #noisy = noisy[:,:64]
        #label = label[:,0:64]
        #print(noisy.size())
        #output =output[:,0:64]
        # print(output.size())
        #vad=np.copy(label[:, 65])
        vad = label.clone()
        vad = vad[:,64]
        vad = vad.reshape((-1, 1))
        gain = label.clone()
        gain = vad*gain[:,:64]
        gain = torch.clamp(gain,max=1)
        floor = 0.05*torch.ones_like(gain)
        one = torch.ones_like(gain)
        gain = torch.where(gain < 0.6, floor, gain)
        gain = torch.log10(gain)
        #gain=np.copy(label[:,0:64])
       # loss = self.loss_kernel(output, gain)
        #print(gain[1,:])
        #mask = 0.8*vad+(1-0.8)*(1-vad)
        # mask = mask.reshape((-1, 1))
        #f_mask=torch.cat((torch.ones(1,38),torch.ones(1,26)),1)
        #f_mask = f_mask.cuda().reshape((1,64))
        #print(f_mask)
        #print(f_mask.shape)
        #snr =  (gain.pow(2)+0.001)/(1-gain.pow(2)+0.001)
        #print(mask)
        #print(vad.shape)
        #loss = (mask*snr*((gain-output).pow(2))).mean()
        #print(gain.shape)
        #loss = (mask*(gain-output).pow(2)).sum()
        loss = ((gain-output).pow(2)).sum() #gain
       #print(loss)
        loss_statistics = self._get_statistics(loss,output, gain, label_len)
        return loss, loss_statistics

    def _get_statistics(self, loss, output, label, label_len):
        loss_item = loss.item()  # utterance leval ctc loss * utterances
        frames = sum(label_len).item()
        loss_statistics = {
            'loss': loss_item,
            'total_frames': frames,
        }

        return loss_statistics

    def log_line(self, reduced_stat):
        
        """Convert the reduced statistics into a log line"""
        #loss_per_frame = reduced_stat['loss'] / reduced_stat['total_frames']/4i5
        #loss_per_frame = reduced_stat['loss']
        loss_per_frame = reduced_stat['loss'] / reduced_stat['total_frames']/64
        return f'Lossperframe: {loss_per_frame:.3f}'
