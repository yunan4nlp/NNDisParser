path=/home/ubuntu/NNDisParser-cp/
cmake .. -DEIGEN3_DIR=${path}eigen -DN3L_DIR=${path}N3LDG -DMKL=TRUE -DCMAKE_BUILD_TYPE=Debug
