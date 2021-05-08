import os

path = 'C:\\Users\\yang\\Desktop'

base_dir = os.path.join(path, '222')
# print(os.listdir(西瓜))

for i, name in enumerate(os.listdir(base_dir)):
    fname = os.path.join(base_dir, name)
    # print(fname)
    new = os.path.join(base_dir, 'weakness_l' + '.' + f'{i}' + '.jpg')

    os.rename(fname, new)