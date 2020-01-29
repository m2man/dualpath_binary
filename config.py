dictionary_path = 'Flickr30k_dictionary.pickle'
dataset_path = 'Flickr30k_dataset.pickle'
numb_epochs = 34
seeds = [1509 + x for x in range(numb_epochs)] # Since stage 1 have 50 epoch
last_epoch = 31
last_index = 2880
batch_size = 40 #increase 32 for each 20 epoch, max is 256 --> need huge amount of GPU RAM
learn_rate = 0.001
moment_val = 0.9

image_folders = ['flickr30k-images-sub-1/', 'flickr30k-images-sub-2/', 
                 'flickr30k-images-sub-3/', 'flickr30k-images-sub-4/',
                 'flickr30k-images-sub-5/']

decay_lr = False
decay_lr_portion = 0.3
decay_lr_min = 0.00005

lamb_1 = 1
stage_2 = False
lamb_0 = 0
alpha = 1

kernel_init = False