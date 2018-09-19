# RST Discourse Parser #

This is the code for the paper : [<strong>Nan Yu</strong>, Meishan Zhang, Guohong Fu. Transition-based Neural RST Parsing with Implicit Syntax
Features. COLING'18, 2018.](http://aclweb.org/anthology/C18-1047)

## How to compile this project in Linux. ##
* Step 0: Open terminal, and change directory to project directory. </br> Use this command  `cd /your/project/path/NNDisParser`. </br>
* Step 1: Create a new directory in NNDisParser. </br>Use this command `mkdir build`.</br>
* Step 2: Change your directory. Use this command `cd build`. </br>
* Step 3: Use this command to build project.</br> `cmake .. -DEIGEN3_DIR=${path}eigen -DN3L_DIR=${path}N3LDG -DMKL=TRUE`. </br>
* Step 4: Now you can compile this project by command `make`. </br>
* Step 5: If you want to run this project. And add this argument. </br>
`-train /your/training/corpus -dev /your/development/corpus -test /your/test/corpus -option /your/option/file -l` </br>


## Data ##
RST Tree Bank.</br>
*https://catalog.ldc.upenn.edu/LDC2002T07*


## External Resource ##
Pretrained word embeddings.</br>
*https://nlp.stanford.edu/projects/glove*


## NOTE ##
Make sure you have eigen ,N3LDG, cmake. </br>
Eigen:*http://eigen.tuxfamily.org/index.php?title=Main_Page*</br>
cmake:*https://cmake.org*</br>

