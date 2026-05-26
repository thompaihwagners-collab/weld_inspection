import os
d = r'F:\ai-tools\weld_inspection\output\dataset'
for s in ['train','val','test']:
    imgs = os.listdir(d + '\\' + s + '\\images')
    print(f'{s}: {len(imgs)} images')
