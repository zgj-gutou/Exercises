import pymongo
import re
from random import sample

client = pymongo.MongoClient('192.168.120.50', 2333)
#建立和数据库系统的连接,创建Connection时，指定host及port参数
db_auth = client.raw
db_auth.authenticate("moresec", "2DWJzdVTe8Jw")
#admin 数据库有帐号，连接-认证-切换库
db = client.raw
print(db)
myCollection = db['zgj']
m = re.compile(u"[\u4e00-\u9fa5]+")
collectionList = ['risk_p1','risk_p2','risk_p3_risk','risk_p4_risk','risk_p5_risk','risk_p6_risk','risk_p8_risk']
List = []
for collectionName in collectionList:
    collection = db[collectionName]  # 选中collection
    results = collection.find()
    count = results.count()
    print(count)
    indexs = sample(range(count),500)
    # df = DataFrame(columns=('handled_content', 'label'))
    for index in indexs:
        # print(result)
        result = results[index]
        source = collectionName
        append_id = result['append_id']
        html=result['html']
        label = result['risk']
        for line in html.split('\n'):
            list = re.findall(m, line)
            List = List+list
        str = ""
        for j in List:
            if str=='':
                str = str+j
            else:
                str = str+","+j
        # s = pd.Series({'handled_content':str, 'label': label})
        # df = df.append(s, ignore_index=True)
        List = []
        myDict = {"source": source, "append_id": append_id, "html": html,"handled_content":str,"lable":label}
        x = myCollection.insert_one(myDict)



