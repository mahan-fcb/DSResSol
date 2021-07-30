import confuse
from lib.config import load_config
import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter
from tensorflow.keras.preprocessing.sequence import pad_sequences


class Util:

    config = None

    char_dict = None

    codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

    def __init__(self):
        self.config = confuse.Configuration('DsResSol', __name__)
        self.config = confuse.Configuration('DsResSol', __name__)
        self.char_dict = self.create_dict(self.codes)
        load_config(self.config, "config.yml")

    def integer_encoding(self, data):
        encode_list = []
        for row in data['Seq'].values:
            row_encode = []
            for code in row:
                row_encode.append(self.char_dict.get(code, 0))
            encode_list.append(np.array(row_encode))

        return encode_list

    def create_dict(self, codes):
        char_dict = {}
        for index, val in enumerate(codes):
            char_dict[val] = index+1
        return char_dict

    def load_file(self):
        df_train = pd.read_pickle(
            self.config['data_files']['df_train'].get())
        df_test1 = pd.read_pickle(
            self.config['data_files']['df_test1'].get())
        dt = pd.read_pickle(self.config['data_files']['dt'].get())
        dk1 = pd.read_pickle(self.config['data_files']['dk1'].get())
        df_test = pd.read_csv(
            self.config['data_files']['df_test'].get())
        dk = pd.read_csv(self.config['data_files']['dk'].get())
        return df_train, df_test1, dt, dk1, df_test, dk

    def normalize(self, data: np.ndarray, scaler=preprocessing.MinMaxScaler()):
        return pd.DataFrame(scaler.fit_transform(data))

    def add_seq_count_attribute(self, data: np.ndarray):
        data['seq_char_count'] = data['Seq'].apply(lambda x: len(x))
        return data

    def get_code_freq(self, data: pd.DataFrame, data_name):
        data = data.apply(lambda x: " ".join(x))

        codes = []
        for i in data:  # concatination of all codes
            codes.extend(i)

        codes_dict = Counter(codes)
        codes_dict.pop(' ')  # removing white space

        data = pd.DataFrame(
            {'Code': list(codes_dict.keys()), 'Freq': list(codes_dict.values())})
        return data.sort_values('Freq', ascending=False).reset_index()[['Code', 'Freq']]

    def pad_data(self, data):
        return pad_sequences(
            data,
            maxlen=self.config['model_config']['max_pad_length'].get(),
            padding='post',
            truncating='post'
        )
