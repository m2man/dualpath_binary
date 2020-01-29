'''
Run evaluation by loss and accuracy (classification) on validate set
'''

from model import *
import mylibrary as mylib
import random
import tqdm
import numpy as np
import pickle
import config
import os
import argparse

cosine_loss = keras.losses.CosineSimilarity()
entropy_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

def eval_model(model, list_dataset):
  accuracy = 0
  correct = 0
  total = 0
  lamb_0 = 1
  alpha = 1
  lamb_1 = 1
  classify_loss = 0
  ranking_loss = 0

  batch_dataset = mylib.generate_batch_dataset_v4(list_dataset, batch_size=32, fill_remain=False)

  for index, batch_eval in enumerate(batch_dataset):
    total += len(batch_eval)
    batch_size = int(len(batch_eval)/2)

    img_ft, txt_ft, lbl = mylib.get_feature_from_batch_v4(batch_eval, 
                                                          image_folders=config.image_folders,
                                                          dictionary=my_dictionary,
                                                          resize_img=224, max_len=32)
    inputs = [tf.convert_to_tensor(img_ft, dtype=tf.float32), 
              tf.convert_to_tensor(txt_ft, dtype=tf.float32)]    

    output, f_image_y, f_text_y = model(inputs)
    classify_loss += lamb_1 * entropy_loss(y_true=lbl, y_pred=output) * batch_size * 2
    
    correct_class = output>0.5
    correct_class = np.where(correct_class)[1]
    correct += np.sum(np.equal(correct_class,lbl))

    ranking_loss = 0
    for i in range(batch_size):
      Ia = f_image_y[i]
      Ta = f_text_y[i]
      In = f_image_y[batch_size + i]
      Tn = f_text_y[batch_size + i]
      ranking_loss += tf.math.add(tf.math.maximum(0.0, alpha - (cosine_loss(Ia, Ta) - cosine_loss(Ia, Tn))), 
                                  tf.math.maximum(0.0, alpha - (cosine_loss(Ta, Ia) - cosine_loss(Ta, In))))
    ranking_loss += lamb_0 * ranking_loss

  classify_loss /= total
  ranking_loss /= (total/2)
  accuracy = correct / total

  return accuracy, classify_loss, ranking_loss

def parse_arguments():
  parser = argparse.ArgumentParser(
      description="Run evaluation on loss and accuracy"
  )

  parser.add_argument(
        "-se",
        "--start_epoch",
        type=int,
        nargs="?",
        help="Choose trained model from specific epoch to run evaluation",
  )

  parser.add_argument(
        "-ee",
        "--end_epoch",
        type=int,
        nargs="?",
        help="Choose trained model from specific epoch to run evaluation",
        default=0
  )

  return parser.parse_args()

def main():
  # Argument parsing
  args = parse_arguments()

  log_filename = os.path.join('report/version_4/','dualpath_v4_stage_1_eval.log')
  if not os.path.exists('report/version_4/'):
    os.mkdir('report/version_4/')

  images_names = list(dataset_flickr30k.keys())

  index_flickr30k = [i for i in range(len(dataset_flickr30k))]
  index_train, index_validate, index_test = mylib.generate_3_subsets(index_flickr30k, ratio = [0.93, 0.035, 0.035])
  print("Number of samples in 3 subsets are {}, {} and {}". format(len(index_train), len(index_validate), len(index_test)))

  images_names_train = [images_names[i] for i in index_train]
  images_names_val = [images_names[i] for i in index_validate]
  images_names_test = [images_names[i] for i in index_test]

  list_dataset = []
  all_class = []
  for idx, img_name in enumerate(images_names_val):
    img_class = img_name[0:-4] # remove '.jpg'
    all_class += [img_class]
    for des in dataset_flickr30k[img_name]:
      temp = [img_name, des, img_class]
      list_dataset.append(temp)

  start_epoch = args.start_epoch
  if args.end_epoch == 0 or args.end_epoch < start_epoch:
    end_epoch = start_epoch
  else:
    end_epoch = args.end_epoch

  for idx in range(start_epoch, end_epoch+1):  
    print("Evaluating Epoch {}".format(idx))
    pretrained_model = 'checkpoints_v4_stage_1/checkpoints_v4_' + str(idx) + '/my_checkpoint'

    model = create_model(dict_length=len(my_dictionary), pretrained_model=pretrained_model, print_summary=False)

    accuracy, class_loss, rank_loss = eval_model(model, list_dataset)

    info = "Epoch: {}\nLoss_Classify: {:.6f}\nLoss_Ranking: {:.6f}\nAccuracy: {:.4f}\n-----".format(idx, class_loss.numpy(), rank_loss.numpy(), accuracy)

    print("Writing to report file")
    with open(log_filename, 'a') as f:
      f.write(info)

if __name__ == "__main__":
  main()