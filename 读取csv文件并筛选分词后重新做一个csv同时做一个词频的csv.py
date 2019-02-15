# for i in range(10):
#     str = ""
#     str = str+"a"
#     print(i)

import pandas as pd
from pandas import DataFrame
import jieba
import string
# collectionList = ['risk_p1','risk_p2','risk_p3_risk','risk_p4_risk','risk_p5_risk','risk_p6_risk','risk_p8_risk']
collectionList = ['test','test1']
wrongList = ['nan','拒绝访问','正在跳转']
newDf = DataFrame(columns=('vocab', 'label'))
dict = {}
for collection in collectionList:
    # dfTemp = pd.read_csv(collection+'.csv',encoding = 'utf-8')
    # df = pd.concat([df,dfTemp])  # df拼接
    df = pd.read_csv(collection+".csv",encoding='utf-8')
    df['handled_content'] = df['handled_content'].fillna("nan")
    for i in range(0, len(df)):  # 遍历
        if(df.iloc[i]['handled_content'] in wrongList):
            # 不要这一行
            continue
        else:
            str = ""
            contents = df.iloc[i]['handled_content'].split(",")
            for content in contents:
                if(len(content) <= 4):
                    if str=="":
                        str = str+content
                    else:
                        str = str+","+content
                    if content in dict:
                        dict[content]+=1
                    else:
                        dict[content]=1
                else:
                    vocabList=jieba.cut(content)
                    for vocab in vocabList:
                        if str == "":
                            str = str + vocab
                        else:
                            str = str + "," + vocab
                        if vocab in dict:
                            dict[vocab] += 1
                        else:
                            dict[vocab] = 1
            s = pd.Series({'vocab': str, 'label': df.iloc[i]['label']})
            newDf = newDf.append(s, ignore_index=True)
    newDf.to_csv(collection + "_useful_items.csv", encoding='utf-8')
    newDf.drop(newDf.index,inplace=True)

dfDict = pd.DataFrame(columns=('key', 'value'))
sortedList = sorted(dict.items(), key=lambda d: d[1],reverse=True)
for i in sortedList:
    seriesDict = pd.Series({'key':i[0],'value':i[1]})
    dfDict = dfDict.append(seriesDict, ignore_index=True)
# print(dfDict)
dfDict.to_csv("dict.csv",encoding ='utf-8')


# a = df.iloc[0]['handled_content'].split(",")
# print(a)
#
# seg_list = jieba.cut("我来到北京清华大学")
# print("Full Mode:", seg_list)  # 全模式
# for i in seg_list:
#     print(i)
#     print(type(i))

