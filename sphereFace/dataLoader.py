import glob
import numpy as np
import os.path as osp
from PIL import Image
import random
from torch.utils.data import Dataset
from matlab_cp2tform import get_similarity_transform_for_PIL

class BatchLoader(Dataset ):
    def __init__(self, imageRoot = '../CASIA-WebFace/',
            alignmentRoot = './data/casia_landmark.txt', cropSize = (96, 112) ):
        super(BatchLoader, self).__init__()

        self.imageRoot = imageRoot
        self.alignmentRoot = alignmentRoot
        self.cropSize = cropSize
        refLandmark = [ [30.2946, 51.6963],[65.5318, 51.5014],
                [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        self.refLandmark = np.array(refLandmark, dtype = np.float32 ).reshape(5, 2)

        with open(alignmentRoot, 'r') as labelIn:
            labels = labelIn.readlines()

        self.imgNames, self.targets, self.landmarks = [], [], []
        for x in labels:
            xParts = x.split('\t')
            self.imgNames.append(osp.join(self.imageRoot, xParts[0] ) )
            self.targets.append(int(xParts[1] ) )
            landmark = []
            for n in range(0, 10):
                landmark.append(float(xParts[n+2] ) )
            landmark = np.array(landmark, dtype=np.float32 )
            self.landmarks.append(landmark.reshape(5, 2) )

        self.count = len(self.imgNames )
        self.perm = list(range(self.count ) )
        random.shuffle(self.perm )

    def __len__(self):
        return self.count

    def __getitem__(self, ind ):

        imgName = self.imgNames[self.perm[ind] ]
        landmark = self.landmarks[self.perm[ind] ]
        target = np.array([self.targets[self.perm[ind] ] ], dtype=np.int64 )

        # Align the image
        img = Image.open(imgName )
        img = self.alignment(img, landmark )
        img = (img.astype(np.float32 ) - 127.5) / 128

        batchDict = {
                'img': img,
                'target': target
                }
        return batchDict


    def alignment(self, img, landmark ):
        tfm = get_similarity_transform_for_PIL(landmark, self.refLandmark.copy() )
        img = img.transform(self.cropSize, Image.AFFINE,
                tfm.reshape(6), resample=Image.BILINEAR)
        img = np.asarray(img )
        if len(img.shape ) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        else:
            img = img[:, :, ::-1]

        img = np.transpose(img, [2, 0, 1] )
        return img

class IdentityLoader():
    """
    Load the images for a given identity.
    """
    def __init__(self, imageRoot = '../CASIA-WebFace/',
            alignmentRoot = './data/casia_landmark.txt', cropSize = (96, 112) ):
        super(IdentityLoader, self).__init__()
        self.imageRoot = imageRoot
        self.alignmentRoot = alignmentRoot
        self.cropSize = cropSize
        refLandmark = [ [30.2946, 51.6963],[65.5318, 51.5014],
                [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
        self.refLandmark = np.array(refLandmark, dtype = np.float32 ).reshape(5, 2)

        with open(alignmentRoot, 'r') as labelIn:
            labels = labelIn.readlines()

        self.identities = {}
        for x in labels:
            xParts = x.split('\t')
            identity = xParts[0].split('/')[0]
            # print(identity)
            landmark = []
            for n in range(0, 10):
                landmark.append(float(xParts[n+2] ) )
            landmark = np.array(landmark, dtype=np.float32 )
            
            tup = self.identities.get(identity)
            if tup:
                tup['img'].append(osp.join(self.imageRoot, xParts[0]))
                tup['target'].append(int(xParts[1]))
                tup['landmark'].append(landmark.reshape(5, 2) )
            
            else:
                tup = {'img': [osp.join(self.imageRoot, xParts[0])],
                       'target': [int(xParts[1])],
                       'landmark': [landmark.reshape(5, 2)]
                      }
                self.identities[identity] = tup
                      
        self.count = len(self.identities.keys())
        self.identities_lst = list(self.identities.items())
        
        self.perm = list(range(self.count))
        random.shuffle(self.perm)
                       
   
    def __len__(self):
        return self.count

    # for enumeration
    def __getitem__(self, ind):

        values = self.identities_lst[self.perm[ind]]# key
        imgNames = values[1]['img']
        targets = values[1]['target']
        landmarks = np.array(values[1]['landmark'], dtype=np.int64 )

        # Align the image
        imgs = []
        for imgName, landmark in zip(imgNames, landmarks):
            img = Image.open(imgName )
            img = self.alignment(img, landmark )
            img = (img.astype(np.float32 ) - 127.5) / 128
            img = np.expand_dims(img, axis=0) # reshape dimension
            imgs.append(img)
        
        batchDict = {
                'img': imgs,
                'target': targets,
                'identity': values[0]
                }
        return batchDict
    
    # for getting a specific identity
    def get(self, identity):
        values = self.identities.get(identity)# key
        if not values:
            print("Identity not in dataset. Identity: %s" % identity)
            return
        
        imgNames = values['img']
        targets = values['target']
        landmarks = np.array(values['landmark'], dtype=np.int64 )

        # Align the image
        imgs = []
        for imgName, landmark in zip(imgNames, landmarks):
            img = Image.open(imgName )
            img = self.alignment(img, landmark )
            img = (img.astype(np.float32 ) - 127.5) / 128
            img = np.expand_dims(img, axis=0) # reshape dimension
            imgs.append(img)
        
        batchDict = {
                'img': imgs,
                'target': targets,
                'identity': identity
                }
        return batchDict
    
    def alignment(self, img, landmark ):
        tfm = get_similarity_transform_for_PIL(landmark, self.refLandmark.copy() )
        img = img.transform(self.cropSize, Image.AFFINE,
                tfm.reshape(6), resample=Image.BILINEAR)
        img = np.asarray(img )
        if len(img.shape ) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        else:
            img = img[:, :, ::-1]

        img = np.transpose(img, [2, 0, 1] )
        return img