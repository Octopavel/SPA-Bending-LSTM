import data_preprocessing as dpp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import min_acc, max_acc, min_analog, max_analog, min_angle, max_angle
# from pandas._libs.tslibs import Timestamp

def load_freehand_datasets():
  return dpp.load_freehand_data()

def clean_freehand_datasets():
  train, valid, test = load_freehand_datasets()
  new_train = []
  new_valid = []
  new_test = []
  # def ranges for training data
  s_range = [700,700,800,800,0]
  e_range = [0,0,440,0,0]
  for i, df in enumerate(train):
    new_train.append(train[i].loc[s_range[i]:df.shape[0]-e_range[i]].reset_index(drop=True))

  # redef ranges for validation data
  s_range = [1315,1200,800,1200,0]
  e_range = [0,0,740,400,0]
  for i, df in enumerate(valid):
    new_valid.append(valid[i].loc[s_range[i]:df.shape[0]-e_range[i]].reset_index(drop=True))

  # redef ranges for testing data
  s_range = [700,800,1000,900,0]
  e_range = [0,500,1100,500,0]
  for i, df in enumerate(valid):
    new_test.append(test[i].loc[s_range[i]:df.shape[0]-e_range[i]].reset_index(drop=True))

  return new_train, new_valid, new_test

def get_features_n_targets(df_list,feature_names = ['acc_x','acc_y','acc_z','pressure'], target_names = ['phi']):
  return dpp.get_features_n_targets(df_list,feature_names,target_names)

def scale_features(df_list):
  for i, df in enumerate(df_list):
    df_list[i].loc[:,'acc_x'] = dpp.scale_data(df['acc_x']*9.81,min_acc,max_acc,-1,1)
    df_list[i].loc[:,'acc_y'] = dpp.scale_data(df['acc_y']*9.81,min_acc,max_acc,-1,1)
    df_list[i].loc[:,'acc_z'] = dpp.scale_data(df['acc_z']*9.81,min_acc,max_acc,-1,1)
    df_list[i].loc[:,'pressure'] = dpp.scale_data(df['pressure'],400,1800,-1,1)
    # df_list[i].loc[:,'flex'] = scale_data(df['flex'],min_analog,max_analog)
  return df_list

def scale_targets(df_list):
  for i, df in enumerate(df_list):
      df_list[i].loc[:,'phi'] = dpp.scale_data(df['phi'],min_angle,max_angle,-1,1)
    # df_list[i].loc[:,'theta'] = scale_data(df['theta'],min_angle,max_angle)
    # df_list[i].loc[:,'psi'] = scale_data(df['psi'],min_angle,max_angle)
  return df_list

def get_scaled_features_n_targets(df_list, downsample = False):
  features,targets = get_features_n_targets(df_list)
  features = scale_features(features)
  targets = scale_targets(targets)
  feat_np = convert_list_df2np(features)
  targ_np = convert_list_df2np(targets)
  if downsample:
    feat_np_new = []
    targ_np_new = []
    for i,_ in enumerate(feat_np):
      feat_np_new.append(downsample_data(feat_np[i]))
      targ_np_new.append(downsample_data(targ_np[i]))
    feat_np = feat_np_new
    targ_np = targ_np_new

  return feat_np,targ_np

def convert_list_df2np(df_list):
  np_list = []
  for df in df_list:
    np_list.append(df.to_numpy())
  return np_list

def downsample_data(data, factor=5):
    # Ensure the length of data is divisible by the factor
    trimmed_length = (len(data) // factor) * factor
    trimmed_data = data[:trimmed_length]
    
    # Reshape the data to introduce a new dimension for averaging
    reshaped_data = trimmed_data.reshape(-1, factor, data.shape[1])
    
    # Compute the mean along the new dimension
    downsampled_data = reshaped_data.mean(axis=1)
    
    return downsampled_data

def moving_average(data, window_size=3):
    # Initialize an array of zeros with the same shape as the input data
    averaged_data = np.zeros_like(data)

    # Loop over each column separately to apply the moving average
    for i in range(data.shape[1]):
        # Compute the moving average using numpy's convolve method
        # We use 'same' mode to ensure the output has the same length as the input
        averaged_data[:, i] = np.convolve(data[:, i], np.ones(window_size)/window_size, mode='same')
    
    return averaged_data

def reshape(np_arr_list):
  out = []
  for data in np_arr_list:
    out.append(data.reshape(data.shape[0],1,data.shape[1]))
  
  return out
  # return dpp.reshape_for_stateful(np_arr_list)
