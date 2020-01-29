'''
Define your functions here
'''
import random
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import array_to_img
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize

# for data augmentation
datagen = ImageDataGenerator(rotation_range=10, 
                            horizontal_flip=True, 
                            width_shift_range=0.05, 
                            height_shift_range=0.05,
                            brightness_range=[0.95,1.02])
                            
ps = PorterStemmer() 
stop_words = stopwords.words('english')
stop_words += [',', '.', '!', '#', '$', '%', '&', '(', ')', 
               '+', '-', '@', '?', '<', '>']

def process_sentence(s):
  s = s.lower()
  s = s.replace("-", " ")
  token = word_tokenize(s)
  token = [ps.stem(x) for x in token if x not in stop_words]
  s = ' '.join(token)
  return s
                 
def generate_3_subsets(sequence, ratio = [0.6, 0.2, 0.2], seed = 159):
  if len(ratio) != 3 or ratio[0] + ratio[1] + ratio[2] != 1:
    print("ratio parameter is not statisfied --> Using default [0.6, 0.2, 0.2]")
    ratio = [0.6, 0.2, 0.2]
  random.Random(seed).shuffle(sequence)
  numb_set = [int(len(sequence) * x) for x in ratio]
  numb_set[-1] = len(sequence) - numb_set[0] - numb_set[1]
  subset_1 = sequence[0:numb_set[0]]
  subset_2 = sequence[numb_set[0] : (numb_set[0] + numb_set[1])]
  subset_3 = sequence[(numb_set[0] + numb_set[1]) : ]
  return subset_1, subset_2, subset_3
  
def generate_augmentation_images(image_path, number_of_images = 5, resize = None, save = './', prefix_name = 'image'):
  # input is a path to the image
  # resize is an integer value
  '''
  datagen = ImageDataGenerator(rotation_range=15, 
                               horizontal_flip=True, 
                               width_shift_range=0.05, 
                               height_shift_range=0.05,
                               brightness_range=[0.95,1.02])
  '''
  datagen = ImageDataGenerator(horizontal_flip=True)
  img = load_img(image_path)
  data = img_to_array(img)
  if resize != None:
    data = cv2.resize(data, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
  save_img(save + prefix_name + '_original.jpg', data)
  samples = np.expand_dims(data, 0)
  it = datagen.flow(samples, batch_size=1)
  for i in range(number_of_images):
    batch = it.next()
    #image = batch[0].astype('uint8')
    aug_img = array_to_img(batch[0])
    save_img(save + prefix_name + '_' + str(i) + '.jpg', aug_img)

def create_images_dataset(link_to_folder, file_specific='.jpg'):
  # Create a 4D numpy array for images (number_of_images x H x W x C)
  # Create the label of that images based on their prefix name (onehot coding)
  list_images = [f for f in os.listdir(link_to_folder) if f.endswith(file_specific)]
  list_images_class = [x[0:(x.find("_"))] for x in list_images]
  onehot_encoded = create_onehot_all_label(list_images_class)
  for index, img in enumerate(list_images):
    data = load_img(link_to_folder + img)
    data = img_to_array(data)
    data = np.expand_dims(data, 0)
    if index == 0:
      result = data
    else:
      result = np.vstack((result, data))
  return result, onehot_encoded

def create_onehot_all_label(list_label):
  # Create onehot coding from the list_label
  # Ex: input = [a, b] --> output = [[1., 0.],[0., 1.]]
  # Also return the labelencoder class inorder to reverse the name (if applicable)
  values = np.array(list_label)
  ## integer encode
  label_encoder = LabelEncoder()
  integer_encoded = label_encoder.fit_transform(values)
  #print(integer_encoded)
  ## binary encode
  onehot_encoder = OneHotEncoder(sparse=False)
  integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
  return label_encoder, onehot_encoded

def create_onehot_label(label, list_label, onehot_all_label):
  # Get the onehot encoding of the label in the list_label and list onehot_label
  index = list_label.index(label)
  result = onehot_all_label[index]
  result = np.expand_dims(result, 0)
  return result

def get_label_from_onehot(label_encoder, onehot_label):
  # Get the name of the class from the onehot encoder label
  return label_encoder.inverse_transform([np.argmax(onehot_label)])[0]

def embedding_image(link_to_image, resize=None, scale=True, prob_aug=0.5):
  # prob_aug is probability that you may want to apply data augmentation(flip, illumination, skew, ...)
  img = load_img(link_to_image)
  data = img_to_array(img)
  if resize != None:
    data = cv2.resize(data, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
  if scale:
    data = data/255.0
  samples = np.expand_dims(data, 0)

  if random.random() < prob_aug:
    it = datagen.flow(samples, batch_size=1)
    samples = it.next()
    
  return samples
  
def embedding_sentence(sentence, dictionary, max_len=32, prob_aug=1):
  # remove words in sentence but not in dictionary
  # create vector size of max_len x number of words in dictionary, in which 1 is at word that appears in dictionary
  # If sentence longer than max_len --> remove some last words
  # If sentence shorter than max_len --> add 0 to the last elements
  # BUT prob_aug is probability that you may want to apply data augmentation
  # This augmentation is randomly at 0 to beginning or the last elements randomly
  sentence = process_sentence(sentence)
  token = word_tokenize(sentence)
  token_in_dict = [x for x in token if x in dictionary]
  len_sentence = len(token_in_dict)
  numb_possible_roll = max_len - len_sentence # for data augmentation (add padding randomly at the begin and end of sentence)
  sentence_embedded = np.zeros([max_len, len(dictionary)])
  for i, tok in enumerate(token_in_dict):
    if i >= max_len:
      break
    sentence_embedded[i, dictionary.index(tok)] = 1

  if numb_possible_roll >= 1 and random.random() < prob_aug:
    row_move = random.randint(0, numb_possible_roll)
    sentence_embedded = np.roll(sentence_embedded, row_move, axis=0)

  sentence_embedded = np.expand_dims(sentence_embedded, 0)
  sentence_embedded = np.expand_dims(sentence_embedded, 0)
  return sentence_embedded

# ===== VERSION 4: 2 classes =====
def generate_batch_dataset_v4(dataset, batch_size=32, seed=159, fill_remain=True):
  # Generate a list in which each of element includes 2*batch_size, in which half is true class, and other half is for ranking loss and class 0
  # version_4 --> binary classification
  dataset_cp = dataset.copy()
  random.Random(seed).shuffle(dataset_cp)
  total_sample = len(dataset_cp)
  list_batch_data = []
  for i in range(total_sample//batch_size):
    batch_data = []
    batch_data += [dataset_cp[i*batch_size + j][:2] + [1] for j in range(batch_size)]
    for j in range(batch_size):
      class_j = dataset_cp[i*batch_size + j][2]

      rand_img_idx = random.randint(0, total_sample-1)
      while(dataset_cp[rand_img_idx][2] == class_j):
        rand_img_idx = random.randint(0, total_sample-1)
      rand_img = dataset_cp[rand_img_idx][0]
          
      rand_txt_idx = random.randint(0, total_sample-1)
      while(dataset_cp[rand_txt_idx][2] == class_j):
        rand_txt_idx = random.randint(0, total_sample-1)
      rand_txt = dataset_cp[rand_txt_idx][1]

      rand_lbl = 0
      batch_data += [[rand_img, rand_txt, rand_lbl]]

    list_batch_data += [batch_data]
  
  # For the remainder
  remainder = total_sample % batch_size
  if remainder > 0:
    if fill_remain:
      missing = batch_size - remainder
    else:
      missing = 0
    batch_data = []
    batch_data += [dataset_cp[i*batch_size + j][:2] + [1] for j in range(remainder)]
    batch_data += [dataset_cp[j] for j in range(missing)]
    for j in range(batch_size):
      if j < remainder:
        class_j = dataset_cp[i*batch_size + j][2]
      else:
        class_j = dataset_cp[j-remainder][2]

      rand_img_idx = random.randint(0, total_sample-1)
      while(dataset_cp[rand_img_idx][2] == class_j):
        rand_img_idx = random.randint(0, total_sample-1)
      rand_img = dataset_cp[rand_img_idx][0]
          
      rand_txt_idx = random.randint(0, total_sample-1)
      while(dataset_cp[rand_txt_idx][2] == class_j):
        rand_txt_idx = random.randint(0, total_sample-1)
      rand_txt = dataset_cp[rand_txt_idx][1]

      rand_lbl = 0
      batch_data += [[rand_img, rand_txt, rand_lbl]]
    list_batch_data += [batch_data]

  return list_batch_data

def get_feature_from_batch_v4(batch_data, image_folders, dictionary, resize_img=224, max_len=32):
  # Get input feature from given batch_data (1 data element in a batch)
  # Output will be img, txt, and label ft
  batch_size = int(len(batch_data) / 2)
  batch_img = np.zeros([len(batch_data), resize_img, resize_img, 3])
  batch_txt = np.zeros([len(batch_data), 1, max_len, len(dictionary)])
  batch_lbl = np.zeros(batch_size * 2)
  batch_lbl[:batch_size] = 1

  for i in range(batch_size):
    img_x = batch_data[i][0]
    try:
      for image_folder in image_folders:
        try:
          image_input_x = embedding_image(link_to_image=image_folder+img_x, resize=224)
          break
        except FileNotFoundError:
          continue
    except:
      print("Error happen in {}!".format(img_x))
    batch_img[i] = image_input_x

    img_x = batch_data[batch_size + i][0]
    try:
      for image_folder in image_folders:
        try:
          image_input_x = embedding_image(link_to_image=image_folder+img_x, resize=224)
          break
        except FileNotFoundError:
          continue
    except:
      print("Error happen in {}!".format(img_x))
    batch_img[batch_size + i] = image_input_x

    text_x = batch_data[i][1]
    text_input_x = embedding_sentence(text_x, dictionary, max_len=max_len)
    batch_txt[i] = text_input_x

    text_x = batch_data[batch_size + i][1]
    text_input_x = embedding_sentence(text_x, dictionary, max_len=max_len)
    batch_txt[batch_size + i] = text_input_x

  return batch_img, batch_txt, batch_lbl