from data.data_load import mem, memnew, memuse
from data.data_load import page
from data.data_load import net, neterror, netpacket, netsize

import pandas as pd

mem.rename(columns={'Memory pcordb02': 'time'}, inplace=True)
memnew.rename(columns={'Memory New pcordb02': 'time'}, inplace=True)
memuse.rename(columns={'Memory Use pcordb02': 'time'}, inplace=True)
page.rename(columns={'Paging pcordb02': 'time'}, inplace=True)
net.rename(columns={'Network I/O pcordb02': 'time'}, inplace=True)
neterror.rename(columns={'Network Errors pcordb02': 'time'}, inplace=True)
netpacket.rename(columns={'Network Packets pcordb02': 'time'}, inplace=True)
netsize.rename(columns={'Network Size pcordb02': 'time'}, inplace=True)


def train_model() -> pd.DataFrame:
    train = mem.merge(memnew, on=['time'], how='outer')
    dataset = [memuse, page, net, neterror, netpacket, netsize]
    for data in dataset:
        train = train.merge(data, on=['time'], how='outer')
    return train
