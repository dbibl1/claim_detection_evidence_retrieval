import pandas as pd


class Claimbuster:
    def __init__(self):
        self.df_cb = pd.read_csv('crowdsourced.csv')

    def transform_df(self):
        df_cb_data = self.df_cb[['Text', 'Verdict']]
        df_cb_data.loc[df_cb_data['Verdict'] < 1, 'Verdict'] = 0
        df_cb_data.rename(columns={'Verdict': 'labels'}, inplace=True)

        return df_cb_data

    def get_df(self):
        return self.transform_df()