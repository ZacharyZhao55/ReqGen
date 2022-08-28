#### 项目结构

​	-exp 五折交叉验证实验

​		-back 存放生成的结构化数据

​		-backEvaluation.py 合成结构化数据，计算识别准确率

​		-dataset 存放五折交叉验证的训练集特征表

​		-pre 存放单数据组的特征表

​		-formaData.py 将特征表整合为五折交叉验证格式，放入dataset

​		-output 存放crf识别的结果

​		-tuple_eval.json 准确率文件

​		-five_group 存放执行文件，生成各元组特征表和运行脚本

​	-gannt 测试集验证

​		-auto_gannt.sh 运行脚本

​		-back 存放生成的结构化数据

​		-generate_tuple.py 合成结构化数据，计算识别准确率

​		-tuple_eval.json 准确率文件

​		-structure 存放crf识别的结果

​		-no_exp 存放执行文件，生成各元组特征表

​	-CRF++-0.58 CRF建模工具

​		-example

​			-exp 五折交叉验证

​				-exec.sh 训练脚本 调用各元组训练脚本

​			-gannt 测试集验证

​	-json_data 存放需求集数据

​		-get_five_data.py 生成五折交叉验证ABCDE数据组，存放于five文件夹下

​		-five 存放数据组 （运行代码时会自动创建）

​	-priority 需求间关联类型的识别+计算需求优先级

​		-gannt 存放gannt数据集识别结果

​		-pure 存放pure数据集识别结果

​		-wmd_relation.py WMD算法识别

​		-graphviz_draw 关系网可视化+计算需求优先级并可视化

​		

#### 运行

##### 基于CRF的结构化算法

**执行文件中需要修改对应位置的绝对地址**

1.修改地址后运行json_data/get_five_data.py 

2.运行 exp/five_group/auto.sh

3.得到五折交叉验证的模型，存放于CRF++-0.58/example/exp/model中，选择效果较好的model进行测试，修改测试脚本中的model地址 (如exec_uav.sh等)

4.运行gannt/auto_gannt.sh

5.通过exp/backEvaluation.py 和 gannt/generate_tuple.py 可以得到验证集和测试集的结构化数据



##### 应用结构化结果识别需求间关联类型

**执行文件中需要修改对应位置的绝对地址(root_path)**

1.修改地址后执行wmd_relation.py得到识别结果

2.运行graphviz_draw.py计算需求优先级并可视化