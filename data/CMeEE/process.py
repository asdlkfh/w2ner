import json

# contents = []
# with open('CMeEE_dev.json','r',encoding='utf-8') as f:
#     content = json.load(f)
# for i in content:
#     if len(list(i['text']))<=200:
#         dic = {}
#         ner = []
#         dic['sentence'] = list(i['text'])
#         for j in i["entities"]:
#             entity = {}
#             index = list(range(j["start_idx"],j["end_idx"]+1))
#             type = j["type"]
#             entity["index"] = index
#             entity["type"] = type
#             ner.append(entity)
#         dic['ner'] = ner
#         contents.append(dic)

# with open('dev.json','w') as f:
#     json.dump(contents,f,ensure_ascii=False)
# print(len(contents))
with open('dev.json','r',encoding='utf-8') as f:
    content = json.load(f)
print(len(content))
test = content[:2500]
dev = content[2500:]
with open('dev.json','w') as f:
    json.dump(dev,f,ensure_ascii=False)

with open('test.json','w') as f:
    json.dump(test,f,ensure_ascii=False)

# with open('test.json','r',encoding='utf-8') as f:
#     content = json.load(f)
# for i in content:
#     for j in i['ner']:
        
# print(len(content))

# contents = [i for i in content if len(i['sentence'])<=512]

# with open('test.json','w') as f:
#     json.dump(contents,f,ensure_ascii=False)
# print(len(contents))  