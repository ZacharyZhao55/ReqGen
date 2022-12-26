# ReqGen

The Github link includes 3 folders：  
bas_dataset: Building Automation System (BAS) dataset, including open source requirement specifications and domain ontologies  
uav_dataset: Unmanned Aerial Vehicle (UAV) dataset, including open source requirement specifications and domain ontologies  
req_gen_unilm: The code of ReqGen 

The requirements of UAV are from the University of Notre Dame, including 99 requirements.   
And the ontology of UAV includes 400 entities.   
The requirements of BAS are from the Standard Building Automation Service (BAS) Specification (2015) consisting of 456 requirements.   
And the open domain model of BAS includes 484 entities.

In each dataset, include 4 files and 1 folder:  
*.src.me.5hop.txt: The knowledge of injection.  
*.src.no.me.txt: The keywords-set of input.  
*.tgt.txt: The target rquirement sentences of input.  
*.structure.json: The semantic roles of keywords for syntax constrained decoding is provided by requirement analysts.  
k_folk: The K-folk data

env install:

  conda create -n unilm python=3.7

  conda activate unilm

  pip install torch==1.7.0

  pip install --user tensorboardX stanza six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools rouge nltk jieba transformers gensim~=3.8.3 Cython>=0.28.5 psutil>=5.6.2 Theano>=0.8.1

  cd Desktop/*_req_gen_unilm_v2/nlg-eval-master

  python setup.py install

  cd ..

  cd unilm/src

  pip install --user --editable .

  cd ../..

model train:

  bash train.sh

inference:

  CUDA_VISIBLE_DEVICES=0,1,2,3 bash test.sh 30 test
  
run k-folk code:

  python k_fold_data_train_test.py


