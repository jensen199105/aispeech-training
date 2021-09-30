class Batch(dict):
    """Batch is a key-value container to hold all the information for one batch

    The Batch is the communication protocol among BatchLoader, Model and Loss.
    """

    def to(self, *args, **kwargs):
        """Automatically convert the payload to desired datatypes

        It's intended for apex.amp, to support transparent FP16 optimization level.
        Only feature matrix will be converted
        """
        batch = self.copy()
        batch['feat'] = self['feat'].to(*args, **kwargs)
        # Allows for batch without 'extra' key
        try:
            batch['extra']['feat'] = [feat.to(*args, **kwargs) for feat in self['extra']['feat']]
        except KeyError:
            pass
        return batch


class Field():
    """An abstract definition for data structure describing a batched tensor of variable lengths

    Args:
        tensor (Tensor): the main payload
        length (LongTensor, optional): the length for each utterance

    """

    def __init__(self, tensor, length=None):
        self.tensor = tensor
        self.length = length

    def __repr__(self):
        return f'Field(tensor={self.tensor}, length={self.length})'

    def to(self, *args, **kwargs):
        """Convert only the tensor to desired dtype

        Please refer to pytorch's document for detailed instruction
        """
        return Field(self.tensor.to(*args, **kwargs), self.length)
