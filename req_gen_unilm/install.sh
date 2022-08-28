conda create -n unilm python=3.7

conda activate unilm

pip install torch==1.7.0

pip install --user tensorboardX six numpy tqdm path.py pandas scikit-learn lmdb pyarrow py-lz4framed methodtools py-rouge pyrouge nltk jieba transformers gensim~=3.8.3 Cython>=0.28.5 psutil>=5.6.2 Theano>=0.8.1

cd Desktop/uav_req_gen_unilm_v2/nlg-eval-master

python setup.py install

cd ..

cd unilm/src

pip install --user --editable .

cd ../..
