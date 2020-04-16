TV News Scene Recognition
=========================
This application is a part of my thesis: Scene type recognition of TV News broadcasts using visual data. Find out more at [Digital Library of University of West Bohemia](https://dspace5.zcu.cz/handle/11025/110/browse?type=author&order=ASC&rpp=20&value=Vysko%C4%8Dil%2C+Ji%C5%99%C3%AD)

Run
---

This section specifies, how to run the main features via the command line (optional arguments are denoted in square brackets, a list of these arguments can be found bellow):

__1. Creating a dataset:__
    
The script expects three input directories (including training, validation, and testing data) and an output path. If any of the inputs is not available (for example, if the created dataset only contains training and validation data), the "None" argument shall be passed (according to the example, the path to the test data must be "None").

Examples:

    python run.py -[-ow] [-v] ./train_data/ ./validation_data/ ./test_data/ ./output/dataset.h5

    python run.py -[-ow] [-v] ./train_data/ ./validation_data/ None ./output/dataset.h5


_Note 1: A folder structure of a data directory (e.g. training_data directory):_

* training_data (directory)

    * Class_1 (directory)
    
        * image_11.jpg
        
        * image_12.jpg
        
            ...
        
        * image_1x.jpg
    
    * Class_2 (directory)
        
        * image_21.jpg
        
        * image_22.jpg
        
            ...
            
        * image_2y.jpg
    
        ...
    
    * Class_n (directory)
        
        * image_n1.jpg
        
        * image_n2.jpg
        
            ...
            
        * image_nz.jpg
       
       
______________
 
_Note 2: A folder structure of a directory with sequences of images (used for time-distributed networks):_

* training_data (directory)

    * Class_1 (directory)
    
        * Input_11 (directory)
    
            * image_111.jpg
            
            * image_112.jpg
            
                ...
            
            * image_11x.jpg
    
        * Input_12 (directory)
    
            * image_121.jpg
            
            * image_122.jpg
            
                ...
            
            * image_12x.jpg
    
            ...
    
        * Input_1i (directory)
    
            * image_1i1.jpg
            
            * image_1i2.jpg
            
                ...
            
            * image_1ix.jpg
            
        ...

    * Class_n (directory)
    
        * Input_n1 (directory)
    
            * image_n11.jpg
            
            * image_n12.jpg
            
                ...
            
            * image_n1x.jpg
    
            ...
    
        * Input_nj (directory)
    
            * image_nj1.jpg
            
            * image_nj2.jpg
            
                ...
            
            * image_njx.jpg
    
__2. Training a network__
    
__2.a Backbone model__

To train a backbone model, -m and -F arguments shall be used.

Example:

    python run.py [-ow] [-v] [-gc] -F -m MODEL_TYPE ./dataset.h5 ./output/

__2.b Time-distributed model__

In case of a time-distributed model, -m argument shall be combined with -w argument, which expects a path to the backbone model that has already been trained before. The -F argument is also required to specify that the model shall be trained. A used dataset must contain sequences of images.

Example

    python run.py [-ow] [-v] [-gc] -F -m MODEL_TYPE -w LOAD_WEIGHTS ./dataset_lstm.h5 ./output/

__3. Evaluation of a trained network__

The trained network (loaded by using -l argument) can be evaluated over the validation and test dataset by using -V or -T arguments (Note: evaluation over the validation dataset is performed automatically by using -F argument, i.e. training the network).

     python run.py [-ow] [-v] [-gc] -V -l LOAD_MODEL ./dataset.h5 ./output/
     
     python run.py [-ow] [-v] [-gc] -T -l LOAD_MODEL ./dataset.h5 ./output/

List of optional arguments
--------------------------

Global:

* -h, --help

    _help message_

* -ow, --overwrite
    
    _overwrite files in the output directory_

* -v, --verbose

    _display extended information while script is running_
    
Specification of the model:

* -m MODEL_TYPE, --model_type MODEL_TYPE

    _create a new model (see src/const_spec.py file, class Architecture)_

* -w LOAD_WEIGHTS, --load_weights LOAD_WEIGHTS

    _load a model, which weights will be used for training the LSTM network (this argument shall be combined with -m argument)_

* -l LOAD_MODEL, --load_model LOAD_MODEL

    _load a model, which will be used for evaluation (-m and -w arguments should not be used)_

Specification of the process:

* -F, --fit

    _train the model_

* -V, --valid

    _generate Confusion matrix and evaluation file (HTML) with predictions for given validation data_

* -T, --test

    _generate Confusion matrix and evaluation file (HTML) with predictions for given test data_

* -P, --predict

    _generate HTML file with predictions_

Optional arguments for the evaluation

* -gc, --gradCAM

    _use Grad-CAM for evaluation (this feature is only supported for networks that are not processing time-series data)_
    