#!/bin/sh
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/restriction_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/first_restriction.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_1110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_1110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/restriction_pre_E.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/restriction_1110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/restriction_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/second_restriction.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_2110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_2110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/restriction_pre_A.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/restriction_2110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/restriction_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/third_restriction.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_3110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_3110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/restriction_pre_B.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/restriction_3110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/restriction_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/four_restriction.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_4110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_4110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/restriction_pre_C.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/restriction_4110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/restriction_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/five_restriction.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_5110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/restriction_model_5110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/restriction_pre_D.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/restriction_5110.output



