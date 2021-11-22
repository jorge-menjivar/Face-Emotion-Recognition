# ECS171-Facial-Recognition
This is the final Project of ECS 171 Fall 2021

# Data set:
## Link to the dataset:
https://drive.google.com/drive/folders/1wFmbLEN46-IVUxRl3lhv3m-EkXKfhp80?usp=sharing

## Brief description:
There are four image sets under ./emotic:
 * ade20k
 * emodb_small
 * framesdb
 * mscoco
 
The dataset can be found in ./Anotations/Annotations.mat
Annotation.mat file can be read using matlab. We can read it in python using:
<pre><code>
import numpy as np
import h5py
f = h5py.File('Annotation.mat','r')
data = f.get('data/variable1')
data = np.array(data) # For converting to a NumPy array
</code></pre>
I will take a look into version compatibility. This way to read .mat file in python is yet to be tested.


# Install the gui python environments
* you need to put
```console
pip install -r requirements.txt
```
in GUI's root entry to install the requirement

