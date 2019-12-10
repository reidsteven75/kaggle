import os
MODE = os.environ['MODE']
print(MODE)

if __name__ == '__main__':
  if (MODE == 'train'):
    import train
  elif (MODE == 'predict'):
    import predict