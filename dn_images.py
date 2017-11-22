import urllib.request
import pandas as pd
import sys
from PIL import Image
import os

def down(url,path):
        f = open(path, 'wb')
        f.write(urllib.request.urlopen(url,timeout=5).read())
        f.close()


set = 'train'
data = pd.read_csv('../data/{}/lim_images.csv'.format(set))
#downloaded = pd.DataFrame(columns=['ImageID','OriginalURL'])
#not_downloaded = pd.DataFrame(columns=['ImageID','OriginalURL'])

for row in data.iterrows():

    imageid = row[1]['ImageID']
    imageurl = row[1]['OriginalURL']
    size = float(row[1]['OriginalSize'])/1000000
    imageformat = imageurl.split('.')[-1]
    path = '../data/{}/images/{}.{}'.format(set,imageid,imageformat)

    try:
        if not os.path.exists(path):
            print ("Downloading: {} of size {} MB".format(path,size))
            down(imageurl,path)
            #urllib.request.urlretrieve(imageurl, path)
            im = Image.open(path)
            im.verify()
#            downloaded.loc[len(downloaded)] = [imageid,imageurl]
#            downloaded.to_csv('downloaded.csv')
            print ("Downloaded and validated: {}".format(path))
            print ("Total images:{}".format(len(os.listdir('../data/{}/images'.format(set)))))

    except Exception as e:
        print (e)
        print ('Image : {} couldnt be downloaded'.format(path))
        if os.path.exists(path):
            os.remove(path)
#        not_downloaded.loc[len(not_downloaded)] = [imageid,imageurl]
#        not_downloaded.to_csv('not_downloaded.csv')

    except (KeyboardInterrupt, SystemExit):

        print ('program terminated, deleting {}'.format(path))
        if os.path.exists(path):
            os.remove(path)
        sys.exit()
