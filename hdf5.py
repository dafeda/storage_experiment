# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Testing writing to $N$ `hdf5` files in parallel where each contains $p$ numbers

# %%
import numpy as np
rng = np.random.default_rng()
import pandas as pd

from p_tqdm import p_map

# %%
# Number of parameters
p = 100000
# Number of ensembles
N = 100


# %%
def write_to_many(N):
    params = pd.DataFrame(rng.normal(size=p), columns=[f"{N}"])
    params.to_hdf(f"Ensemble_{N}.h5", key="parameters", mode="a")
    responses = pd.DataFrame(rng.normal(size=p), columns=[f"{N}"])
    responses.to_hdf(f"Ensemble_{N}.h5", key="responses", mode="a")
    return True


# %% [markdown]
# ## Write to files

# %%
# %%time
res = p_map(write_to_many, np.arange(0, N))
print(f"Writing to {N} hdf5 files where each contains {p} numbers takes:")

# %% [markdown]
# ## Read from files

# %%
# %%time
df = pd.DataFrame(index=np.arange(0, p))
for i in range(N):
    df = df.join(pd.read_hdf(f"Ensemble_{i}.h5", key="parameters"))
    
print(f"Combining {N} hdf5 files containing {p} numbers each into one data frame takes:")

# %%
df.shape

# %%
assert df.shape == (p, N)

# %% [markdown]
# ## Clean up

# %%
from pathlib import Path
for p in Path(".").glob("Ensemble_*.h5"):
    p.unlink()

# %%
