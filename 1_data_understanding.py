import data
from utils import fix_randomness

import yaml

# read config file
with open("./config/config.yaml", "r") as ymlfile:
    try:
        cfg = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)

problem = cfg['problem']
datapath = cfg["dataset"][problem]["path_to_data"]

if problem == 'covid19':
    data.covid19_data_look(datapath)
elif problem == 'adults':
    data.adults_data_look(datapath)
elif problem == 'fivethirtyeight':
    data.fivethirtyeight_data_look(datapath)
elif problem == 'gss':
    data.gss_data_look(datapath)
else:
    print("The problem was not supported!!!")