#商铺数据处理
import pandas

data = pandas.read_csv('D://002-python-minig- weizhuanye//data.csv',encoding='utf-8')
#解析数据，存成列表形式

dataD = data.to_dict(orient='records')
#数据的清洗

for i in data.index:
    data.comment[i] = data.comment[i].split('                    ')[0]

for i in data.index:
    data.price[i] = data.price[i].split('                    ')[-1][1:]
#去除字段缺失的数据        
for i in data.index:
    if ('我要点评' in data.comment[i]) | ('-' in data.price[i]):
        data = data.drop(i)
    
data2 = data.dropna(axis=0)
#将commentlist拆成三分字段，并且清洗成数字
data3 = pandas.DataFrame(columns=['口味','环境','服务'])
data['口味'] = None
data['环境'] = None
data['服务'] = None
for i in data.index:
    data['质量'][i] = data.commentlist[i].split('                                ')[0][2:]
    data['环境'][i] = data.commentlist[i].split('                                ')[1][2:]
    data['服务'][i] = data.commentlist[i].split('                                ')[2][2:]
data = data.drop(['commentlist'],axis = 1)

import pickle
pic = open(r'D:\002-python-minig- weizhuanye\data.pkl','wb')
pickle.dump(data,pic)
pic.close()