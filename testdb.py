import sqlite3
import getdatafile as gdata
import pandas as pd

conn = sqlite3.connect("tempDB.db")
# df2M = gdata.get_dataframe_from_label('healthy', 1)
# df2M['shaft speed'] = (4 * df2M['shaft speed'] * 10**5) / 1024
# df2M.to_sql("df2M", conn, if_exists= 'replace')
df2M = pd.read_sql("SELECT [shaft speed], [vibration velocity] from df2M;", conn)
print(df2M.info())