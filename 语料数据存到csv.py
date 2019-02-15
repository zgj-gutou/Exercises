import pymongo
import re
import pandas as pd
from pandas import DataFrame

# myclient = pymongo.MongoClient("mongodb://192.168.120.50:2333/",username="moresec",password="2DWJzdVTe8Jw")
# myclient = pymongo.MongoClient(host='192.168.120.50', port=2333,username="moresec",password="2DWJzdVTe8Jw")
# myclient = pymongo.MongoClient(host="192.168.120.50:2333",username="moresec",password="2DWJzdVTe8Jw")

client = pymongo.MongoClient('192.168.120.50', 2333)
#建立和数据库系统的连接,创建Connection时，指定host及port参数
db_auth = client.raw
db_auth.authenticate("moresec", "2DWJzdVTe8Jw")
#admin 数据库有帐号，连接-认证-切换库
db = client.raw
print(db)
m = re.compile(u"[\u4e00-\u9fa5]+")
collectionList = ['risk_p1','risk_p2','risk_p3_risk','risk_p4_risk','risk_p5_risk','risk_p6_risk','risk_p8_risk']
# for i in collectionList:
collection = db['risk_p1']
results = collection.find()
# print(results.count())
List = []
df = DataFrame(columns=('original_content','handled_content', 'label'))
i=0
for result in results:
    # print(result)
    f=result['html']
    label = result['risk']
    for line in f.split('\n'):
        list = re.findall(m, line)
        # for i in list:
        #     print(i)
        List = List+list
    # print(List)
    # print(label)
    # print("----------")
    str = ""
    for j in List:
        if str=='':
            str = str+j
        else:
            str = str+","+j
    s = pd.Series({'handled_content':str, 'label': label})
    df = df.append(s, ignore_index=True)
    List = []
    i=i+1
    if(i==100):
        break
finalFile = df.to_csv("test.csv")



