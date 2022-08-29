import json

# contents = []
# with open('dev.json','r',encoding='utf-8') as f:
#     content = json.load(f)

# for i in content:
#     i.pop('word')
#     contents.append(i)
    
# with open('dev_new.json','w') as f:
#     json.dump(contents,f,ensure_ascii=False)
with open('train.json','r',encoding='utf-8') as f:
    content = json.load(f)
for i in content:
    if len(i['sentence'])>100:
        print(len(i['sentence']))