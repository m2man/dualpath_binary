import tensorflow as tf
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from tensorflow import keras
from tensorflow.keras.regularizers import l2
import numpy as np
import pickle
import config

cosine_loss = keras.losses.CosineSimilarity()
entropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

# ===== LOAD DATA =====
with open(config.dictionary_path, 'rb') as f:
  my_dictionary = pickle.load(f)

with open(config.dataset_path, 'rb') as f:
  dataset_flickr30k = pickle.load(f)

# ===== LOAD KERNEL INIT =====
from gensim.models import KeyedVectors
model_word = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

NUMB_WORDS = len(my_dictionary)
NUMB_FT = 300
kernel_init = np.zeros((NUMB_WORDS, NUMB_FT))
for idx, word in enumerate(my_dictionary):
  try:
    word_2_vec_ft = model_word.word_vec(word)
    word_2_vec_ft = np.reshape(word_2_vec_ft, (1, NUMB_FT))
  except KeyError:
    word_2_vec_ft = np.random.rand(1, NUMB_FT)
  kernel_init[idx,:] = word_2_vec_ft

def kernel_initialization(shape, dtype=None):
    kernel = np.zeros(shape)
    kernel[0,0,:,:] = kernel_init
    return kernel

# ===== DEFINE BRANCHES AND BLOCKS =====    
class BottleNeckResidualBlock(keras.layers.Layer):
  def __init__(self, n_bottleneckfilters, n_filters, kernel_regular = None, downsampling = None):
    super(BottleNeckResidualBlock, self).__init__()
    self.n_filters = n_filters
    self.n_bottleneckfilters = n_bottleneckfilters
    self.kernel_regular = kernel_regular
    self.downsampling = downsampling

  def build(self, input_shape):
    self.projection_or_not = (int(input_shape[-1]) != self.n_filters) or self.downsampling
    
    first_strides = [1, 1] 
    if self.downsampling:
      first_strides = [1, 2]

    self.main_conv1 = keras.layers.Convolution2D(filters=self.n_bottleneckfilters,
                                    kernel_size=[1, 1],
                                    strides=first_strides,
                                    padding='same',
                                    kernel_regularizer=self.kernel_regular,
                                    use_bias=False,
                                    activation=None)
    self.batch1 = keras.layers.BatchNormalization()
    self.relu = keras.layers.ReLU()
    self.main_conv2 = keras.layers.Convolution2D(filters=self.n_bottleneckfilters,
                                    kernel_size=[1, 2],
                                    strides=[1, 1],
                                    padding='same',
                                    kernel_regularizer=self.kernel_regular,
                                    use_bias=False,
                                    activation=None)
    self.batch2 = keras.layers.BatchNormalization()    
    self.main_conv3 = keras.layers.Convolution2D(filters=self.n_filters,
                                    kernel_size=[1, 1],
                                    strides=[1, 1],
                                    padding='same',
                                    kernel_regularizer=self.kernel_regular,
                                    use_bias=False,
                                    activation=None)
    self.batch3 = keras.layers.BatchNormalization()
    
    if self.projection_or_not == True:
      self.project_conv = keras.layers.Convolution2D(filters=self.n_filters,
                                    kernel_size=[1, 1],
                                    strides=first_strides,
                                    padding='same',
                                    kernel_regularizer=self.kernel_regular,
                                    use_bias=False,
                                    activation=None)
      self.project_batch = keras.layers.BatchNormalization()

  def call(self, inputs):
    main_path = self.main_conv1(inputs)
    main_path = self.batch1(main_path)
    main_path = self.relu(main_path)
    main_path = self.main_conv2(main_path)
    main_path = self.batch2(main_path)
    main_path = self.relu(main_path)
    main_path = self.main_conv3(main_path)
    main_path = self.batch3(main_path)
    if self.projection_or_not == True:
      shorcut = self.project_batch(self.project_conv(inputs))
    else:
      shorcut = inputs
    final = main_path + shorcut
    final = self.relu(final)
    return final

class Deep_CNN_Text_Model(keras.Model):
  def __init__(self, l2_rate, kernel_init=True):
    super(Deep_CNN_Text_Model, self).__init__()
    self.l2_rate = l2_rate
    if kernel_init:
      self.block1 = keras.Sequential([
                          keras.layers.Convolution2D(filters=300,
                                              kernel_size=[1,1],
                                              kernel_initializer=kernel_initialization,
                                              strides=[1,1],
                                              padding='same',
                                              kernel_regularizer=l2(self.l2_rate),
                                              use_bias=False,
                                              activation=None),
                          keras.layers.BatchNormalization(),
                          keras.layers.ReLU()            
      ])
    else:
      self.block1 = keras.Sequential([
                          keras.layers.Convolution2D(filters=300,
                                              kernel_size=[1,1],
                                              strides=[1,1],
                                              padding='same',
                                              kernel_regularizer=l2(self.l2_rate),
                                              use_bias=False,
                                              activation=None),
                          keras.layers.BatchNormalization(),
                          keras.layers.ReLU()            
      ])
    self.block2 = keras.Sequential([
                        #keras.layers.MaxPool2D(pool_size=[1,2], strides=[1,2]),
                        BottleNeckResidualBlock(64, 256, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(64, 256, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(64, 256, kernel_regular=l2(self.l2_rate), downsampling=False),      
    ])
    self.block3 = keras.Sequential([
                        BottleNeckResidualBlock(128, 512, kernel_regular=l2(self.l2_rate), downsampling=True),
                        BottleNeckResidualBlock(128, 512, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(128, 512, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(128, 512, kernel_regular=l2(self.l2_rate), downsampling=False),
    ])
    self.block4 = keras.Sequential([
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=True),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(256, 1024, kernel_regular=l2(self.l2_rate), downsampling=False),
    ])
    self.block5 = keras.Sequential([
                        BottleNeckResidualBlock(512, 2048, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(512, 2048, kernel_regular=l2(self.l2_rate), downsampling=False),
                        BottleNeckResidualBlock(512, 2048, kernel_regular=l2(self.l2_rate), downsampling=False),
    ])
    self.block6 = keras.Sequential([
                        keras.layers.GlobalAveragePooling2D(),
                        keras.layers.Dense(2048),
                        keras.layers.BatchNormalization(),
                        keras.layers.ReLU(),
                        keras.layers.Dropout(0.5),
                        keras.layers.Dense(2048),   
    ])

  def call(self, inputs):
    x = self.block1(inputs)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.block6(x)
    return x

# ===== DEFINE LOSS =====
def total_loss(model, input_x, target_y, alpha=1, lamb_0=1, lamb_1=1):
  output, f_image_y, f_text_y = model(input_x)
  classify_loss = lamb_1 * entropy_loss(y_true=target_y, y_pred=output)
  batch_size = int(len(target_y)/2)
  
  ranking_loss = 0
  for i in range(batch_size):
    Ia = f_image_y[i]
    Ta = f_text_y[i]
    In = f_image_y[batch_size + i]
    Tn = f_text_y[batch_size + i]
    ranking_loss += tf.math.add(tf.math.maximum(0.0, alpha - (cosine_loss(Ia, Ta) - cosine_loss(Ia, Tn))), 
                                tf.math.maximum(0.0, alpha - (cosine_loss(Ta, Ia) - cosine_loss(Ta, In))))
  ranking_loss = lamb_0 * ranking_loss / batch_size
  
  loss = tf.math.add(ranking_loss, classify_loss)
  return loss, classify_loss, ranking_loss

# ===== DEFINE GRAD =====
def grad(model, input_x, target_y, alpha=1, lamb_0=1, lamb_1=1):
  with tf.GradientTape() as tape:
    loss, classify_loss, ranking_loss = total_loss(model, input_x, target_y, alpha, lamb_0, lamb_1)
  return loss, classify_loss, ranking_loss, tape.gradient(loss, model.trainable_variables)

# ===== DEFINE ENTIRE MODEL =====
def create_model(dict_length=len(my_dictionary), pretrained_model='', ft_resnet=False, print_summary=True):
  INPUT_SIZE = (224, 224, 3)
  input_test = keras.layers.Input(shape=INPUT_SIZE)
  resnet = keras.applications.ResNet50(input_shape=INPUT_SIZE,
                                      weights='imagenet',
                                      include_top=False)
  resnet.trainable = ft_resnet
  average_pooling = keras.layers.GlobalAveragePooling2D()
  fc_1 = keras.layers.Dense(2048)
  bn_1 = keras.layers.BatchNormalization()
  relu_1 = keras.layers.ReLU()
  do_1 = keras.layers.Dropout(0.5)
  fc_2 = keras.layers.Dense(2048)
  deep_cnn_branch = keras.Sequential([resnet, average_pooling, fc_1, bn_1, relu_1,
                                      do_1, fc_2], name = 'deep_cnn_branch')
  


  deep_text_branch = Deep_CNN_Text_Model(l2_rate = 0.001, kernel_init=config.kernel_init)

  # ----- TRANSFER -----
  l2_rate = 0.001

  # Define Layer
  share_weights_tf = keras.layers.Dense(1024, name = 'share_weights')

  t_bn_share = keras.layers.BatchNormalization()
  t_relu_share = keras.layers.ReLU()
  t_fc_1 = keras.layers.Dense(1024, kernel_regularizer=l2(l2_rate))
  t_bn_1 = keras.layers.BatchNormalization()
  t_relu_1 = keras.layers.ReLU()
  t_block = keras.Sequential([t_bn_share, t_relu_share, t_fc_1, t_bn_1, t_relu_1], name = 'text_after_share')

  i_bn_share = keras.layers.BatchNormalization()
  i_relu_share = keras.layers.ReLU()
  i_fc_1 = keras.layers.Dense(1024, kernel_regularizer=l2(l2_rate))
  i_bn_1 = keras.layers.BatchNormalization()
  i_relu_1 = keras.layers.ReLU()
  i_block = keras.Sequential([i_bn_share, i_relu_share, i_fc_1, i_bn_1, i_relu_1], name = 'image_after_share')

  c_fc_1 = keras.layers.Dense(512, kernel_regularizer=l2(l2_rate))
  c_bn_1 = keras.layers.BatchNormalization()
  c_relu_1 = keras.layers.ReLU()
  c_do_1 = keras.layers.Dropout(0.5)

  c_fc_2 = keras.layers.Dense(256, kernel_regularizer=l2(l2_rate))
  c_bn_2 = keras.layers.BatchNormalization()
  c_relu_2 = keras.layers.ReLU()
  c_do_2 = keras.layers.Dropout(0.5)

  c_fc_3 = keras.layers.Dense(128, kernel_regularizer=l2(l2_rate))
  c_bn_3 = keras.layers.BatchNormalization()
  c_relu_3 = keras.layers.ReLU()
  c_do_3 = keras.layers.Dropout(0.5)

  #c_fc_4= keras.layers.Dense(2) # now its logit --> use loss with from_logit = True (automatic use softmax)
  c_fc_4= keras.layers.Dense(2, activation = 'softmax') # now its prob --> from_logit = False 

  block_1 = keras.Sequential([c_fc_1, c_bn_1, c_relu_1, c_do_1], name = 'fc_512')
  block_2 = keras.Sequential([c_fc_2, c_bn_2, c_relu_2, c_do_2], name = 'fc_256')
  block_3 = keras.Sequential([c_fc_3, c_bn_3, c_relu_3, c_do_3], name = 'fc_128')

  # Function API
  image_input = keras.Input(shape=(224, 224, 3), name='image')  # Variable-length sequence of ints
  text_input = keras.Input(shape=(1, 32, dict_length), name='text') 
  image_f = deep_cnn_branch(image_input)
  text_f = deep_text_branch(text_input)

  image_after_share = share_weights_tf(image_f)
  text_after_share = share_weights_tf(text_f)

  image_after_share = i_block(image_after_share)
  text_after_share = t_block(text_after_share)

  combined_ft = tf.math.multiply(image_after_share, text_after_share, name = 'element_mult')
  combined_ft = block_1(combined_ft)
  combined_ft = block_2(combined_ft)
  combined_ft = block_3(combined_ft)
  out_ft = c_fc_4(combined_ft)

  dualpath_model_tf = keras.Model(inputs=[image_input, text_input], outputs=[out_ft, image_after_share, text_after_share])
  if print_summary:
    dualpath_model_tf.summary()

  if pretrained_model != '':
    # Restore the weights
    dualpath_model_tf.load_weights(pretrained_model)
  else:
    print("No pretrained model given --> Train from scratch")

  return dualpath_model_tf