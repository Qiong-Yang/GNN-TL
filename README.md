# GNN-TL

The GNN-TL can  predict the retention time with the GNN-learned representations. It takes molecular graph as the input, and the predicted retention time as the output. The Overview of GNN-TL is showed as following:

![GNN-TL](https://raw.githubusercontent.com/Qiong-Yang/GNN-TL/main/support/GNN-TL.png)

# Motivation

The combination of spectra matching, retention time and accurate mass can improve the structural annotation in untargeted metabolomics. However, comparatively less attention has been afforded to using retention time in identifying metabolites. This may be caused by the limitation of available retention time data, especially for hydrophilic interaction liquid chromatography (HILIC). Hence, we present the GNN-TL method, which is proved to be an effective way to predict molecular HILIC retention time and improve the accuracy of structural identification.

# Depends

Anaconda for python 3.6

conda install pytorch

conda install -c rdkit rdkit

# Usage

If you want to make the prediction of Fiehnlab HILIC experimental system RT of unknown molecule, please put your spectra files in to **data** directory and run  preprocess.py and transferlearning.py
