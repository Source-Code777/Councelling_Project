import pandas as pd

class Data:
    def __init__(self, url_23: str, url_24: str):
        self.url_23 = url_23
        self.url_24 = url_24

    def load_data(self):
        tables_23 = pd.read_html(self.url_23)
        tables_24 = pd.read_html(self.url_24)

        tables_23 = tables_23[0]
        tables_24 = tables_24[0]

        df=pd.concat([tables_23, tables_24],axis=0, ignore_index=True)

        return df