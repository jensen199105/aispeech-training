import torch
import torch.nn as nn

from .loss import Loss, add_loss

@add_loss('mses')
class MSESLoss(Loss):
    """A warpper of original pytorch CELoss

    It simply unpacks the output and batch into specific format, and computes
    some useful metrics for logging purposes.

    Args:
        output (tensor, B,T,C)
        data_batch (dict)
    """

    def __init__(self, reduction='sum'):
        super().__init__()
        self.loss_mse = nn.MSELoss(reduction=reduction)
        self.loss_bce = nn.BCELoss(reduction=reduction)
    

    def forward(self, output, data_batch):
        if 'skipped_label' in data_batch:
            label_field = data_batch['skipped_label']
        else:
            label_field = data_batch['label']
        label = label_field.tensor
         
        label_len = label_field.length
        '''
        if 'skipped_feat' in data_batch:
            feat_field = data_batch['skipped_feat']
        else:
            feat_field = data_batch['feat']
        
        feat = feat_field.tensor
        '''
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
        label = label.cuda().reshape(-1,label.shape[-1])
        
        '''
        #first loss : the output is gain and loss use mse    
        vad = label.clone()
        vad = vad[:,64]
        vad = vad.reshape((-1, 1))
        label = label[:,:64]
        label = torch.clamp(label,max=1)
        #label = torch.log10(torch.clamp(label,min=0.001)) 
        #label = np.copy(label[:,0:64])
        mask = 0.8*vad+(1-0.2)*(1-vad)
        #f_mask=torch.cat((5*torch.ones(1,38),torch.ones(1,26)),1)
        #f_mask = f_mask.cuda().reshape((1,64))
        #loss = self.loss_mse(output, label)
        output = output[:,:64]
        loss = mask*((label-output).pow(2)).sum()
        

        '''
        '''
        #second loss : the output is gain and vad use bce
        output = torch.clamp(output,min=0.0001,max=0.99999)
        vad = label.clone()
        vad = vad[:,64]
        vad = vad.reshape((-1, 1))
        label = vad*label[:,:65]
        zero = torch.zeros_like(label)
        one = torch.ones_like(label)
        label = torch.where(label < 0.6, zero, one)
        loss = self.loss_bce(output, label)
        '''       
 
        
        # third loss : the output is gain and vad ,gain use mse ,vad use bce
        #vadout = output.clone()
        #vadout = vadout[:,64]
        #vadout = torch.clamp(vadout,min=0.0001,max=0.99999) # if out activations is sigmod ,this step delect
        gainout = output.clone()
        gainout =gainout[:,:64]
        #vad = label.clone()
        #vad = vad[:,64]
        #vad = vad.reshape((-1, 1))
        gain = label.clone()
        gain = torch.clamp(gain,max=1)
        gain = gain[:,:64]
        zero = torch.zeros_like(gain)
        one = torch.ones_like(gain)
        #gain = torch.where(gain < 0.6, zero, gain) #one /gain
        #gain = torch.where(gain < 0.3, zero, gain) #one /gain
        #gain = torch.where(gain >= 0.7, one, gain) #one /gain
        #gain[:,:30] = torch.where(gain[:,:30] < 0.3, zero[:,:30], gain[:,:30])
        gain[:,:30] = torch.where(gain[:,:30] > 0.3, one[:,:30], gain[:,:30])
        gain[:,30:64] = torch.where(gain[:,30:64] < 0.4, zero[:,30:64], gain[:,30:64])
	#gain[:,30:64] = torch.where(gain[:,30:64] > 0.5, one[:,:30], gain[:,30:64])
       
        #gainloss = ((gain-gainout).pow(2)).sum()
        
        low_gainloss = self.loss_mse(gainout[:,:45], gain[:,:45])
        high_gainloss = self.loss_mse(gainout[:,45:], gain[:,45:])
        #vadloss = self.loss_bce(vadout, vad)
        #loss = 2*gainloss+vadloss
        loss = low_gainloss * 5 + high_gainloss
        
         
        loss_statistics = self._get_statistics(loss,output, label, label_len)
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
        loss_per_frame = reduced_stat['loss']/reduced_stat['total_frames']/65 
        return f'Lossperframe: {loss_per_frame:.3f}'
