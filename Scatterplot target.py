import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def grouped_scatter_plot_with_conf(df, column_to_group, target_col, log_scale=False, alpha_level = 1, clip_count = 0, bins_for_cut_count=0, n_std_upper = 2, n_std_lower=1, fill_b_std=True, annotate_the_outliers=True):
  """
  Returns a scatterplot of a column of interest ('column to group') and usually the target column ('target_col') and customizable confidence intervals.

  df: Pandas DataFrame
  column_to_group, target_col: column names from df. column_to_group is a feature which relationship with the target variable is to be reviewed. 
  log_scale: Boolean, default = False - plot x axis on a log scale
  alpha_level: Float - Alpha level of the scatter points
  clip_count: int, default = 0. Allows to remove items from the dataset with less obeservation. For example, clip_count=30 removes the For example, clip_count=30 removes all observations with less than 30 occurrences. 
  bins_for_cut_count: int, default = 0. Cuts the target count column in n number of bins. Impacts the fit of the confidence intervals - the higher n, the higher the overfitting
  n_std_upper, n_std_lower: Float. Number of st.dev. above/below the mean of the confidence intervals.
  fill_b_std, Boolean, default = True. If true, fills the area between the two standard deviations
  annotate_the_outliers, Boolean, default = True. If true, annotates the outliers above the defined boundaries. 
  """
  
  object_to_plot = df.groupby(column_to_group).agg({target_col: ['mean', 'count']}).sort_values(by=(target_col,  'count'), ascending = False)
  target_mean_col = f'{target_col}_mean'
  target_count_col = f'{target_col}_count'
  object_to_plot.columns = [target_mean_col, target_count_col]
  object_to_plot = object_to_plot.loc[object_to_plot[target_count_col]>clip_count, :] #Cut part of the data with low count values

  #Create the lower and upper boundaries
  object_to_plot['qcut_bins'] = pd.qcut(object_to_plot[target_count_col],
                        q=bins_for_cut_count, duplicates='drop')
  object_to_plot['qcut_bins_std'] = object_to_plot.groupby("qcut_bins")[target_mean_col].transform('std')
  object_to_plot['qcut_bins_mean'] = object_to_plot.groupby("qcut_bins")[target_mean_col].transform('mean')


  #Plot the scatterplot
  object_to_plot.plot.scatter(target_count_col, target_mean_col, figsize=((10,8)),logx=log_scale, alpha=alpha_level)
  plt.title(f'Scatter plot of count of {column_to_group} vs. {column_to_group} mean value')
  plt.ylabel(f'Mean {target_col}')
  if log_scale:
    plt.xlabel(f'Log of searches per {column_to_group}')
  else:
    plt.xlabel(f'Searches per {column_to_group}')
 
  #Plot the Mean, lower and upper boundaries 
  plt.plot(object_to_plot[target_count_col], object_to_plot['qcut_bins_mean']+ n_std_upper*object_to_plot['qcut_bins_std'], alpha=1)
  plt.plot(object_to_plot[target_count_col], object_to_plot['qcut_bins_mean'], alpha=1)
  plt.plot(object_to_plot[target_count_col], (object_to_plot['qcut_bins_mean'] - n_std_lower*object_to_plot['qcut_bins_std']).clip(0), alpha=1, color='C0')
  
  if fill_b_std:
    plt.fill_between(object_to_plot[target_count_col], 
                     object_to_plot['qcut_bins_mean']+ n_std_upper*object_to_plot['qcut_bins_std'], 
                     (object_to_plot['qcut_bins_mean'] - n_std_lower*object_to_plot['qcut_bins_std']).clip(0), 
                     alpha=0.1)
    
  #Annotating outliers
  if annotate_the_outliers:
    condition_above = object_to_plot[target_mean_col] > object_to_plot['qcut_bins_mean']+ n_std_upper*object_to_plot['qcut_bins_std']
    condition_below = object_to_plot[target_mean_col] < (object_to_plot['qcut_bins_mean'] - n_std_lower*object_to_plot['qcut_bins_std']).clip(0)
    object_for_annotation = object_to_plot[(condition_above) | (condition_below)][[target_count_col, target_mean_col]]
  
    for k, v in object_for_annotation.iterrows():
      plt.annotate(k, v)

  plt.show()