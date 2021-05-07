import os

path = 'C:\\Users\\yang\\Desktop\\nongzuowu\\datasets'

base_dir = os.path.join(path, 'weakness')
# print(os.listdir(西瓜))

for i, name in enumerate(os.listdir(base_dir)):
    fname = os.path.join(base_dir, name)
    # print(fname)
    new = os.path.join(base_dir, 'weakness' + '.' + f'{i}' + '.jpg')

    os.rename(fname, new)