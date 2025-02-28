#%%
import pandas as pd
from src.utils import PATH

nettows = pd.read_parquet(PATH.PROCESSED_DATA / "nettows_processed.parquet")

# %%
nettows.head()

# %%
nettows.info()

# %%
