import pandas as pd

products = pd.read_csv('data/products.csv')
labelids = products.labelid.values.tolist()

set = 'train'


print('Label IDS fetched')

product_bbox = pd.read_csv('../data/{}/annotations-human-bbox.csv'.format(set))
product_bbox = product_bbox[product_bbox['LabelName'].isin(labelids)]
image_ids = product_bbox.ImageID.unique().tolist()
product_bbox.to_csv('../data/{}/product_bbox.csv'.format(set))

print ('Image IDS fetched and product bbox saved')



for i,chunk in enumerate(pd.read_csv('../data/{}/images.csv'.format(set), chunksize=900000)):
    if i == 0:
        images = chunk[chunk['ImageID'].isin(image_ids)]
    if i > 0:
        images = images.append(chunk[chunk['ImageID'].isin(image_ids)])
    chunk.to_csv('../data/chunk/chunk{}.csv'.format(i))
    print ('chunk{}.csv processed'.format(i))

images.to_csv('../data/{}/product_images.csv'.format(set))

print ('IMAGE data stored to product_images.csv')

lim_imgs = []
for labelid in labelids:
        #getting only upto 500 images for each label
    lim_imgs.extend(product_bbox[product_bbox['LabelName']==labelid].ImageID.unique()[:500])

lim_imgs = list(set(lim_imgs))

limited_images = images[images['ImageID'].isin(lim_imgs)]

limited_images.to_csv('../data/{}/lim_images.csv'.set)

print ('LIMITED IMAGE data stored to lim_images.csv')
