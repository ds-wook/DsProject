import os
import glob
import pandas as pd
from typing import List


def get_merged_csv(flist: List, **kwargs) -> pd.DataFrame:
    '''
    # 각 폴더 별로 정리한 파일을 하나의 csv 파일 형태로 묶음
    - 27개의 항목을 가지고 진행
    - 파일을 합치는 함수를 만들어 호출
    - 27개의 csv 폴더를 만든 후 결측치 확인
    '''
    return pd.concat([pd.read_csv(f, **kwargs) for f in flist],
                     ignore_index=True)


path = '../../project_data'
path_list = os.listdir(path)

for p in path_list:
    fmask = os.path.join(path + "\\" + p, p + "_*.csv")
    globals()[p.lower()] = get_merged_csv(glob.glob(fmask))
