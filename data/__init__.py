from torch.utils.data import DataLoader

from .preprocessing import get_data_to_buffer
from .dataset import BufferDataset
from .collator import collate_fn_tensor


def get_dataloader(train_config, *args, **kwargs):
    buffer = get_data_to_buffer(train_config)
    dataset = BufferDataset(buffer)
    collate_fn = lambda x: collate_fn_tensor(x, train_config)
    dataloader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=0,
    )
    return dataloader
