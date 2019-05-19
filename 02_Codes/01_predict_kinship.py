import dlib 
from pathlib import Path
import pandas as pd
import os
import numpy as np

DATA_DIR = Path(__file__).parent.parent/"01_Data"
FACE_REC_MODEL = DATA_DIR/"dlib_models"/"dlib_face_recognition_resnet_model_v1.dat"
PREDICTOR = DATA_DIR/"dlib_models"/"shape_predictor_5_face_landmarks.dat"

# Relationship data
df_kinship = pd.read_csv(DATA_DIR/"train_relationships.csv")

# Function to extract 128 vector for a given image
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(str(PREDICTOR))
facerec = dlib.face_recognition_model_v1(str(FACE_REC_MODEL))

def img_2_128vec(img):
    """ Takes an image and retuns the 128 dimension vector
    """
    dets = detector(img, 1)
    if len(dets) == 1:
        shape = sp(img, dets[0])
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        return face_descriptor
    else:
        print("More than one faces in the image")
        return -1


# Construct all positive examples. 128 vecs of two related person are appened to make 
# 1 vector of length 256. This is done exhaustively. So if person 1 is realated to person2 
# and they have N and M images respectively, we will get N*M  positive examples. 
all_positive_exs = np.empty((0, 2*128))

for index, row in df_kinship.iterrows():
    
    print(f"{index} out of {df_kinship.shape[0]}")
    dir_p1 = DATA_DIR/"train"/row['p1']
    dir_p2 = DATA_DIR/"train"/row['p2']
    
    # Skip is the folders do not exit
    if not os.path.exists(dir_p1) or not os.path.exists(dir_p2):
        print(f"Data from {index} does not exist")
        continue
        
    for f1 in os.listdir(dir_p1):
        img_1 = dlib.load_rgb_image(str(dir_p1/f1))
        f1_128_vec = img_2_128vec(img_1)
        
        if f1_128_vec == -1:
            print(f'{dir_p1/f1} failed to process')
            continue
        
        for f2 in os.listdir(dir_p2):
            img_2 = dlib.load_rgb_image(str(dir_p2/f2))
            f2_128_vec = img_2_128vec(img_2)
            
            if f2_128_vec == -1:
                print(f"{dir_p2/f2} failes to process")
                continue
            
            # Flatten and join 128 vec of both images
            both_imgs = np.append(f1_128_vec, f2_128_vec)
            # Append to all_positive_exs
            all_positive_exs = np.append(all_positive_exs, both_imgs.reshape((-1, 2*128)), axis=0)            
   
# 158 ,989 examples of kinshiop vectors
np.save(DATA_DIR.parent/"03_Processed"/"all_positive_examples.npy", all_positive_exs )

# Construct negative examples
