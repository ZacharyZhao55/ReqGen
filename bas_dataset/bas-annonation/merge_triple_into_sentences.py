import wordninja

sentense_list = []

result = []
set_result = []
fi_path  = 'bas.src.me.new.5hop.train.txt'
fi = open(fi_path,'w')
with open('bas.src.me.5hop.train.txt') as triple_fi:
    for i in triple_fi.readlines():
        if len(sentense_list) < 10:
            i_list = i.split()
            for w in i_list:
                if w not in sentense_list:
                    if len(w) < 4:
                        pass
                    else:
                        winja = wordninja.split(w)
                        for wj in winja:
                            # if wj not in set_result:
                            set_result.append(wj)
                            sentense_list.append(wj)
        else:
            sentense_list = set(sentense_list)
            result.append(sentense_list)
            sentense_list =[]

for i in result:
    fi.write(' '.join(i))
    fi.write('\n')
