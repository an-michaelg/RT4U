import pandas as pd
import numpy as np
import warnings

from typing import List, Dict, Union  # , Optional, Callable, Iterable
from scipy.io import loadmat

label_schemes: Dict[str, Dict[str, Union[int, float]]] = {
    'tufts':  {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 2},
    'binary': {'normal': 0, 'mild': 1, 'moderate': 1, 'severe': 1},
    'all': {'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3},
    'not_severe': {'normal': 0, 'mild': 0, 'moderate': 0, 'severe': 1},
    'as_only': {'mild': 0, 'moderate': 1, 'severe': 2},
    'mild_moderate': {'mild': 0, 'moderate': 1},
    'moderate_severe': {'moderate': 0, 'severe': 1}
}

class_labels: Dict[str, List[str]] = {
    'tufts': ['No AS', 'Early AS', 'Significant AS'],
    'binary': ['Normal', 'AS'],
    'all': ['Normal', 'Mild', 'Moderate', 'Severe'],
    'not_severe': ['Not Severe', 'Severe'],
    'as_only': ['Mild', 'Moderate', 'Severe'],
    'mild_moderate': ['Mild', 'Moderate'],
    'moderate_severe': ['Moderate', 'Severe']
}


# for now, this only gets the matlab array loader
# Dict is a lookup table and can be expanded to add mpeg loader, etc.
# returns a function
def get_loader(loader_name):
    loader_lookup_table = {'mat_loader': mat_loader}
    return loader_lookup_table[loader_name]


def mat_loader(path):
    return loadmat(path)['cine']


def categories(x, categories_dict):  # TODO check to learn
    # reject nan values
    if pd.isnull(x):
        return np.nan
    text = x.lower()
    for k in categories_dict.keys():
        if k in text:  # try each matching string
            return categories_dict[k]
    # no matches, return the default option
    return categories_dict['other']


def apply_categorical_mapping(pdseries, categories_dict):
    if 'other' not in categories_dict.keys():
        # 'other' behaves differently for boolean vs string
        test_value = list(categories_dict.values())[0]
        if isinstance(test_value, str):
            categories_dict['other'] = 'other'
        elif isinstance(test_value, bool):
            categories_dict['other'] = False
        else:
            raise ValueError('Categorical mappings only support boolean or string arguments')
    return pdseries.apply(lambda x: categories(x, categories_dict))


def get_risk_factors(df):
    """
    analyze risk factors and save as booleans in new columns
    """
    smoking_categories = {'quit': True, 'smoker': True, 'never smoked': False}
    df['Smoked'] = apply_categorical_mapping(df['Risk factors'], smoking_categories)
    df['BloodPressure'] = apply_categorical_mapping(df['Risk factors'], {'high bp': True})
    df['HighCholesterol'] = apply_categorical_mapping(df['Risk factors'], {'high cholesterol': True})
    df['Diabetes'] = apply_categorical_mapping(df['Risk factors'], {'diabetes': True})
    df['Calcified'] = apply_categorical_mapping(df['AV structure'], {'calcified': True})
    return df

def compute_intervals(df, unit, quantity, if_offset=False):
    """
    Calculates the number of sub-videos from each video in the dataset
    Saves the frame window for each sub-video in a separate sheet

    Parameters
    ----------
    df : pd.DataFrame
        dataframe object containing frame rate, heart rate, etc.
    unit : str
        unit for interval retrieval, image/second/cycle
    quantity : 
        quantity for interval retrieval, 
        eg. 1.3 with "cycle" means each interval should be 1.3 cycles

    Returns
    -------
    df : pd.DataFrame
        updated dataframe containing num_intervals and window_size
    df_intervals: pd.DataFrame
        dataframe containing mapping between videos and window start/end frames
        
    """
    ms = df['frame_time']
    hr = df['heart_rate']
    if unit == 'image':
        if int(quantity) < 1:
            raise ValueError('Must draw >= 1 image per video')
        df['window_size'] = int(quantity)
    elif unit == 'second':
        df['window_size'] = (quantity * 1000 / ms).astype('int32')
    elif unit == 'cycle':
        df['window_size'] = (quantity * 60000 / ms / hr).astype('int32')
    else:
        raise ValueError(f'Unit should be image/second/cycle, got {unit}')
    # if there are any window sizes of zero or less, raise an exception
    if len(df[df['window_size']<1]) > 0:
        # TODO pinpoint the videos that cause this
        raise Exception("Dataloader: Detected proposed window size of 0, exiting")
        
    df['num_intervals'] = (df['frames'] / df['window_size']).astype('int32')
    df['leftover_frames'] = (df['frames'] - df['num_intervals'] * df['window_size']).astype('int32')
    
    video_idx, interval_idx, start_frame, end_frame = [], [], [], []
    for i in range(len(df)):
        video_info = df.iloc[i]
        if video_info['num_intervals'] == 0:
            video_idx.append(i)
            interval_idx.append(0)
            start_frame.append(0)
            end_frame.append(video_info['frames'])
        else:
            n_intervals = video_info['num_intervals']
            w_size = video_info['window_size']
            if video_info['leftover_frames'] > 0 and if_offset:
                offset = np.random.randint(video_info['leftover_frames'])
            else:
                offset = 0
            for j in range(n_intervals):
                video_idx.append(i)
                interval_idx.append(j)
                start_frame.append(j * w_size + offset)
                end_frame.append((j+1) * w_size + offset)
    d = {'video_idx':video_idx, 'interval_idx':interval_idx, 
         'start_frame':start_frame, 'end_frame':end_frame}
    df_interval = pd.DataFrame.from_dict(d)
    
    return df, df_interval
