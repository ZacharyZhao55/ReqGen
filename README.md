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
k_folk: The K-fold data
