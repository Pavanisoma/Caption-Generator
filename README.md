# README #

This README would normally document whatever steps are necessary to get your application up and running.

### Flickr 8r dataset ###

Please download the image dataset and description for training here:  
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip  
https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip  

The dataset is up to 1GB, so we only keep the copy of it on the local machine. Please do not push the dataset to the remote repository. I have added the dataset to .gitignore to prevent accidentially push.
Unzip Flickr8k_Dataset.zip into the folder named Flicker8k_Dataset. Unzip Flickr8k_text.zip into the folder named Flickr8k_text. Put these 2 folders inside CaptionGenerator.

> (tf) thy@ironman:~/workspace/cmpe258-teamproject/CaptionGenerator$ ls  
> CaptionGenerator_Mid-reviewPresentation_Demo.ipynb  Flicker8k_Dataset  
> encoded_train_images_vgg16.p                        Flickr8k_text  

### Glove word vectors ###

Please download the glove word vectors for training here: 
https://github.com/stanfordnlp/GloVe
under Download pre-trained word vectors: Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 300d vectors, 822 MB download)
http://nlp.stanford.edu/data/wordvecs/glove.6B.zip  

The dataset is up to 1GB, so we only keep the copy of it on the local machine. Extract the zip file and place the glove folder under source folder. Make sure glove.txt are accesable under the location : 'glove/glove.6B.300d.txt'. This will allow code to read the glove pretrained word vector files.

### Conda - environment set up ###

I highly recommend you to set up virtual environment for this project, that will help you keep the environment clean and not conflicted with other packages required in other projects.   
Tutorial: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html   
For example: I need to create a new virtual env named cmpe_258, and I will install the required packages for the project:
~~~~ 
conda create -n cmpe258_env python=3.6.8
conda activate cmpe258_env
conda install pandas matplotlib jupyter notebook scipy scikit-learn nb_conda nltk spyder pillow nltk glob2
conda install -c conda-forge tensorflow keras
~~~~
You can deactivate the current virtual env by:
~~~~
source deactivate
~~~~
Rember to activate your virtual env before open jupyter notebook:
~~~~
source activate cmpe258_env
jupyter notebook
~~~~
Some other notices:    
- Tensorlow for CPU might not be installed succesfully with python 3.7. So make sure you use python 3.6.8 (that's why we need virtual env, in case you use python 3.7 for other projects)    
- I pushed my packages list in the requirements.txt. But that one has specified packages for GPU (like tensorflow-gpu), so please disregard that.    
- We might need to install more packages on the go, make sure you let the team know which packages are required for your work    
- Try to use conda instead of pip, I know some packages installed by pip might be outdated when working with tensorflow    

### Image Encoder ###
I have finished extracting features from image training set using VGG16 (pre-trained model). After this step, each image can be presented by a vector of 2048 dimensions. This step requires a lot of computation, there's no need to run it all over again. So I dump the result to a file named encoded_train_images_vgg16.p.
You can try to run the function extract_features_vgg16 in the notebook to see how it works. But next time you can ignore it unless you need to extract features for other image datasets.  

### Some packages needed for the project
- conda install pillow
- conda install -c conda-forge pydotplus
- Install ray https://ray.readthedocs.io/en/latest/installation.html (might not need to install ray if you don't re-run the loading input for models)
- Install graphviz (Ubuntu): sudo apt-get install graphviz libgraphviz-dev (optional)
- Install graphviz (MAC) brew install graphviz (optional)


### How to run the prototype scripts on your machine:
- Pull the latest code
- The needed script files are: utils.py, image_encoding.py, model.py, sequence_utils.py, caption_generation.py, model_no_ray.py, caption_generation_no_ray.py
- Train the network and compute the output for sample image: python caption_generation_no_ray.py

### utils.py   
This script has the utils function to load the raw data sets, do some pre-processing data.    
### image_encoding.py
This script has the functions related to image encoding using vgg16.   
### sequence_utils.py   
This script has the functions to extract the bag of words from the Flicker descriptions data, build the tokenizer. Pretrained Glove word vectors are used to extract the embedding matrix, that is given as weights for the Keras Embedding Layer in the neural network.
### model.py or model_no_ray.py   
This script defines the model, loading and aggregating the needed inputs for training the model. model_no_ray.py does not have the function to load input data from scratch.   
### caption_generation.py or caption_generation_no_ray.py   
This script has the main flow of the application, trains the neural network, does the argmax search to output the caption.    

### Must read:
https://towardsdatascience.com/image-captioning-with-keras-teaching-computers-to-describe-pictures-c88a46a311b8   
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/    
These tutorial has the implementation for your tasks.    
