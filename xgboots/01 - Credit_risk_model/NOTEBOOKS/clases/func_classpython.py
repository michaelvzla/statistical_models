import time 
import pandas as pd
import numpy as np
from IPython.display import display
from pathlib import Path
from time import strftime, localtime
from os import listdir
from os.path import isfile, join
import pickle 
import subprocess


class DataAnalyzer:

    def get_na(self, df):
        qsna = df.shape[0] - df.isnull().sum(axis=0)  # Cantidad sin NA
        qna = df.isnull().sum(axis=0)  # Cantidad de NA
        ppna = round(100 * (qna / df.shape[0]), 2)  # Porcentaje NA
        aux = {'Datos sin NAs en Qtd': qsna, 'NAs en Qtd': qna, 'NAs en %': ppna}
        na = pd.DataFrame(data=aux).reset_index()
        na = na.rename(columns={'index': 'Variables'})
        return na.sort_values(by='NAs en %', ascending=False)

class DataSummary:
    def summary(self, df, columna):
        df_filtered = df[columna].value_counts().reset_index()
        df_filtered = df_filtered.rename(columns={'index': 'Categorias', columna: 'Qtd'})
        df_filtered['%'] = round(df_filtered['Qtd'] / df_filtered['Qtd'].sum(), 1) * 100
        return df_filtered

class SummaryTableFormatter:
    def __init__(self, train_data):
        self.train_data = train_data
        self.summary_table = None
        self.min_ne_rate = None
        self.max_ne_rate = None

    def generate_summary_table(self):
        self.train_data['decil'] = pd.qcut(self.train_data['score'], q=10, labels=False)
        
        self.summary_table = self.train_data.groupby('decil').agg({
            'score': ['min', 'max'],
            'resultado_del_riesgo': ['sum', lambda x: len(x) - sum(x)],
        })
        
        self.summary_table.columns = ['Min_score', 'Max_score', 'Eventos', 'No_Eventos']
        
        self.summary_table['Event_Rate'] = self.summary_table['Eventos'] / (
                self.summary_table['Eventos'] + self.summary_table['No_Eventos'])
        self.summary_table['NoEvent_Rate'] = 1 - self.summary_table['Event_Rate']
        self.summary_table = self.summary_table.sort_values(by='decil', ascending=False)
        self.summary_table['Event_dist'] = round(self.summary_table['Eventos'] / self.summary_table['Eventos'].sum(), 2)
        self.summary_table['NoEvent_dist'] = round(
            self.summary_table['No_Eventos'] / self.summary_table['No_Eventos'].sum(), 2)
        self.summary_table['Cumulative_Event_Rate'] = (self.summary_table['Eventos'].cumsum() / self.summary_table[
            'Eventos'].sum())
        self.summary_table['Cumulative_NoEvent_Rate'] = (self.summary_table['No_Eventos'].cumsum() /
                                                         self.summary_table['No_Eventos'].sum())
        
        self.summary_table['KS'] = np.abs(self.summary_table['Cumulative_Event_Rate'] -
                                          self.summary_table['Cumulative_NoEvent_Rate'])
        self.summary_table['Odds'] = self.summary_table['Eventos'] / self.summary_table['No_Eventos']
        
        self.min_ne_rate = self.summary_table['NoEvent_Rate'].min()
        self.max_ne_rate = self.summary_table['NoEvent_Rate'].max()

    @staticmethod
    def format_color(value, min_value, max_value):
        normalized_value = (value - min_value) / (max_value - min_value)
        if normalized_value > 0.6:
            red = 255
            green = 0
        elif 0.4 <= normalized_value <= 0.6:
            red = 255
            green = 255
        else:
            red = 0
            green = 255
        blue = 0
        return f'background-color: rgb({red}, {green}, {blue}); font-weight: normal; color: black'

    def format_percent(self, value):
        return f'{value:.2%}'

    @staticmethod
    def format_int_thousands(value):
        return f'{int(value * 1):,.0f}'

    @staticmethod
    def format_decimal(value):
        return f'{int(value * 100):.2f}'

    @staticmethod
    def format_decimal_ks(value):
        return f'{value:.4f}'

    @staticmethod
    def highlight_max_ks(s):
        max_index = s.idxmax()
        color_styles = ['background-color: lightgreen; font-weight: bold; color: black'
                        if idx == max_index else '' for idx in s.index]
        return color_styles

    def format_table(self):
        formatted_table = self.summary_table.style.applymap(lambda x: self.format_color(x, self.min_ne_rate,
                                                                                        self.max_ne_rate),
                                                            subset=['NoEvent_Rate'])

        formatted_table = formatted_table.format({
            'Min_score': self.format_int_thousands,
            'Max_score': self.format_int_thousands,
            'Event_Rate': self.format_percent,
            'NoEvent_Rate': self.format_percent,
            'Event_dist': self.format_percent,
            'NoEvent_dist': self.format_percent,
            'Cumulative_Event_Rate': self.format_percent,
            'Cumulative_NoEvent_Rate': self.format_percent,
            'KS': self.format_percent,
            'Odds': self.format_decimal_ks
        })

        formatted_table = formatted_table.apply(self.highlight_max_ks, subset=['KS'])

        return formatted_table



