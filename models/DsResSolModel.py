import numpy as np

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold


class DsResSolModel:
    util = None

    def __init__(self, util):
        self.util = util

    def prepare(self):
        #TODO:  df_train, dt: sequences and biological features fo traning set, df_test, dk: sequence and biological features for NESG, test set df_test1, dk1:sequence  and    biological features for Zhang test set 
        df_train, df_test1, dt, dk1, df_test, dk = self.util.load_file()

        #TODO: normalize the biological features for both test sets and training set
        dt = self.util.normalize(dt, MinMaxScaler())
        dk = self.util.normalize(dk, MinMaxScaler())
        dk1 = self.util.normalize(dk1, MinMaxScaler())

     
        X = dt.to_numpy()
        X1 = dk.to_numpy()
        X2 = dk1.to_numpy()

        #TODO: calculate sequence length
        df_train = self.util.add_seq_count_attribute(df_train)
        df_test = self.util.add_seq_count_attribute(df_test)
        df_test1 = self.util.add_seq_count_attribute(df_test1)

   
        classes = df_train['solubility'].value_counts()[:2].index.tolist()

        #TODO: comments
        train_sm = df_train.loc[df_train['solubility'].isin(
            classes)].reset_index()
        test_sm = df_test.loc[df_test['solubility'].isin(
            classes)].reset_index()
        test_sm1 = df_test1.loc[df_test1['solubility'].isin(
            classes)].reset_index()

        #TODO: integer encoding for amino acids in protein sequences
        train_encode = self.util.integer_encoding(train_sm)
        test_encode = self.util.integer_encoding(test_sm)
        test_encode1 = self.util.integer_encoding(test_sm1)

        #TODO: Padding the sequence
        train_pad = self.util.pad_data(train_encode)
        test_pad = self.util.pad_data(test_encode)
        test_pad1 = self.util.pad_data(test_encode1)

        #TODO: comments
        train_ohe = to_categorical(train_pad)
        test_ohe = to_categorical(test_pad)
        test_ohe1 = to_categorical(test_pad1)

        #TODO: comments
        le = LabelEncoder()
        y_train_le = le.fit_transform(train_sm['solubility'])
        y_test_le = le.fit_transform(test_sm['solubility'])
        y_test_le1 = le.fit_transform(test_sm1['solubility'])

        #TODO: comments
        y_train = to_categorical(y_train_le)
        y_test = to_categorical(y_test_le)
        y_test1 = to_categorical(y_test_le1)

        #TODO: comments
        VALIDATION_ACCURACY = []
        VALIDATION_LOSS = []

        #TODO: b, b1 save accuracy of each cross fold on NESG test set and zhang test set respectively. t, t1 save f-1 score of each cross fold on NESG test set and zhang test set respectively
        t = np.zeros(self.util.config['model_config']['kfold'].get(int))
        b = np.zeros(self.util.config['model_config']['kfold'].get(int))
        t1 = np.zeros(self.util.config['model_config']['kfold'].get(int))
        b1 = np.zeros(self.util.config['model_config']['kfold'].get(int))


        #TODO: comments
        skf = StratifiedKFold(n_splits=self.util.config['model_config']['kfold'].get(int),
                              shuffle=True,
                              random_state=self.util.config['model_config']['random_state'].get(int))

        prepared_data = {
            'train_pad': train_pad,
            'test_pad': test_pad,
            'test_pad1': test_pad1,
            'y_train': y_train,
            'y_train_le': y_train_le,
            'y_test': y_test,
            'y_test1': y_test1,
            'skf': skf,
            't': t,
            'b': b,
            't1': t1,
            'b1': b1,
            'X': X,
            'X1': X1,
            'X2': X2,
            'train_ohe': train_ohe
        }
        return prepared_data
