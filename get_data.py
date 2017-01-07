#!/usr/bin/env python

import pandas as pd 
import numpy as np 

def import_data(filename):
	return pd.DataFrame.from_csv(filename, index_col=None)
