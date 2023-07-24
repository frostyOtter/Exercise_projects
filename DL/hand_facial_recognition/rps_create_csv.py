import os
import pandas as pd

def create_csv(img_path):
    # get the target name
    label_list = os.listdir(img_path)
    # create a list to store image name and label
    ImageId = []
    target = []
    # loop over target list
    for label_name in label_list:
        # get the path to each folder
        target_path = os.path.join(img_path, label_name)
        # list all image in that folder
        target_list = os.listdir(target_path)
        # append image name and label to list created
        ImageId.extend(target_list)
        target.extend([label_name]*len(target_list))
    
    # make a dictionary out of 2 list
    df_dict =  {
        'ImageId': ImageId,
        'target': target
    }
    df = pd.DataFrame(df_dict)
    df = df.reset_index(drop = True)
    # return a csv file
    return df.to_csv('/Users/mike/SUMMER2023/DPL302m/RPS_project/rps_dataset.csv')