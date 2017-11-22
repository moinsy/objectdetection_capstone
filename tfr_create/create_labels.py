import pandas as pd

pdts_path = '../data/products.csv'
label_map_path = '../data/label_map.pbtxt'
pdts = pd.read_csv(pdts_path)

with open(label_map_path, 'w+') as f:
    for row in pdts.iterrows():
        id = row[1]['id']
        name = row[1]['labelid']
        display_name = row[1]['labelname']
        line = 'item {\n\tid: '+ str(id) +'\n\tname: '+ name +'\n\tdisplay_name: '+ display_name +'\n}\n'
        f.write(line)