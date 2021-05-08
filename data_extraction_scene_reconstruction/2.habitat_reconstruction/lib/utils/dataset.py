import os
import numpy as np
import torch.utils.data
from lib.utils.io import load


class DatasetRGBD(torch.utils.data.Dataset):


    def __init__(self, root, datafile, cb_load_sample=None, cb_preprocessing=None):

        if datafile not in ["data", "train", "valid", "test"]:
            raise NotImplementedError()

        data = np.loadtxt(os.path.join(root, datafile + ".txt"), dtype=str)

        self.fnames_rgb = np.asarray([os.path.join(root, "rgb", x) for x in data[:,0]])
        self.fnames_depth = np.asarray([os.path.join(root, "depth", x) for x in data[:,0]])
        self.targets = data[:,1].astype(np.int32)
        self.nSamples = data.shape[0]
        self.nClasses = len(set(self.targets))
        self.cb_load_sample = cb_load_sample
        self.cb_preprocessing = cb_preprocessing


    def __getitem__(self, index):

        x_rgb = self.fnames_rgb[index]
        x_depth = self.fnames_depth[index]
        y = self.targets[index]

        if self.cb_load_sample is not None:
            rgb, depth = self.cb_load_sample(x_rgb, x_depth)

        if self.cb_preprocessing is not None:
            rgb, depth, y = self.cb_preprocessing(rgb, depth, y)

        return rgb, depth, y


    def __len__(self):
        return self.nSamples


    def shuffle(self):
        idx = np.random.permutation(self.nSamples)
        self.fnames_rgb = self.fnames_rgb[idx]
        self.fnames_depth = self.fnames_depth[idx]
        self.targets = self.targets[idx]


class Dataset(torch.utils.data.Dataset):
    """
    """


    def __init__(self, root, datafolder, datafile, cb_load_sample=None, cb_preprocessing=None):
        """
        Initialize the dataset.
        The file "root/datafile.txt" will be loaded.
        It contains a list of pairs (filepath, target) where "filepath"
        is the path of the image file to load and "target" is the
        class index associated to that file.
        In the folder "root/dsname" there is also a "label.txt" file
        with the names of all the classes.

        Parameters
        ----------
        root : string
            Data folder containing the dataset
        datafolder : string
            Subfolder in which data is contained with the path structure
            defined by 'datafile'
        datafile : string
            Portion of the data to consider; must be "data", "train", "valid" or "test"
        cb_load_sample : fn (default: None)
            Function taking as parameter the filename of the sample to load
        cb_preprocessing : fn (default: None)
            Function to preprocess the sample loaded through "cb_load_sample"
        """

        if datafile not in ["data", "train", "valid", "test"]:
            raise NotImplementedError()

        data = np.loadtxt(os.path.join(root, datafile + ".txt"), dtype=str)

        self.filenames = np.asarray([os.path.join(root, datafolder, x) for x in data[:,0]])
        self.targets = data[:,1].astype(np.int32)
        self.nSamples = len(self.filenames)
        self.nClasses = len(set(self.targets))
        self.cb_load_sample = cb_load_sample
        self.cb_preprocessing = cb_preprocessing


    def __getitem__(self, index):
        """
        Return a sample and its target values.

        Parameters
        ----------
        index : integer
            Position of the item to extract

        Returns
        -------
        x : depends on cb_load_sample and cb_preprocessing
        y : depends on cb_load_sample and cb_preprocessing
        """

        x = self.filenames[index]
        y = self.targets[index]

        if self.cb_load_sample is not None:
            x = self.cb_load_sample(x)

        if self.cb_preprocessing is not None:
            x, y = self.cb_preprocessing(x, y)

        return x, y


    def __len__(self):
        """
        Get the length of the dataset in terms of number of samples.

        Returns
        -------
        nSamples : integer
            Number of elements in the dataset
        """
        return self.nSamples


    def shuffle(self):
        """
        In place random permutation of the dataset.
        """
        idx = np.random.permutation(self.nSamples)
        self.filenames = self.filenames[idx]
        self.targets = self.targets[idx]


class SIXD:
    """
    Wrapper for SIXD datasets
    """

    def __init__(self, prefix):
        """
        Ctor
        Initializes the wrapper by providing a prefix to the dataset

        Parameters
        ----------
        prefix : string
            Data folder containing the dataset
        """
        self.models = os.path.join(prefix, "models", "obj_{:02d}.ply")


    def model(self, cid):
        """
        Retrieve CAD model info

        Parameters
        ----------
        cid : int
            Class id

        Returns
        -------
        data : dictionary
            Ply data (dictionary) with keys "vertices", "normals", "colors" and "faces"
        """
        return load(self.models.format(cid))
