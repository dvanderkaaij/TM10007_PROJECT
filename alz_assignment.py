'''
Docstring
'''

# Data loading functions. Uncomment the one you want to use
from adni.load_data import load_data
# from brats.load_data import load_data
# from hn.load_data import load_data

data = load_data()
print(f'The number of samples: {len(data.index)}')
print(f'The number of columns: {len(data.columns)}')