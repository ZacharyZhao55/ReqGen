#!/bin/sh
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/agent_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/first_agent.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_1110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_1110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/agent_pre_E.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/agent_1110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/agent_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/second_agent.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_2110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_2110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/agent_pre_A.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/agent_2110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/agent_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/third_agent.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_3110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_3110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/agent_pre_B.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/agent_3110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/agent_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/four_agent.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_4110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_4110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/agent_pre_C.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/agent_4110.output

/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_learn -c 10.0 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/template/agent_template /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/five_agent.txt /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_5110
/root/Desktop/crf_project/crf_project/CRF++-0.58/crf_test  -m /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/model/agent_model_5110 /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/pre_data/agent_pre_D.txt > /root/Desktop/crf_project/crf_project/CRF++-0.58/example/exp/output/agent_5110.output




