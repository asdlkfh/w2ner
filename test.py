import argparse
import json
import numpy as np
import prettytable as pt
import torch
import torch.autograd
import torch.nn as nn
import transformers
from sklearn.metrics import precision_recall_fscore_support, f1_score
from torch.utils.data import DataLoader

import config
import test_data_loader
from test_data_loader import LoadTestDataBert
import utils
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./config/resume-zh.json')
args = parser.parse_args()
config = config.Config(args)



class SinglePredict:

    def __init__(self,device,model_path):
        self.path = model_path
        self.setdevice(device)
        # self.data_loader,self.ori_data = self.dataload(data)
        self.loadtestdatabert = LoadTestDataBert(config)
        self.model = Model(config)
        self.model.to(device)
        self.model.load_state_dict(torch.load(self.path))

    def setdevice(self,device):
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{device}")
        else:
            device = torch.device('cpu')

    def dataload(self,data):
        self.datasets, self.ori_data = self.loadtestdatabert.load_test_data_bert(data)  #输出构造的数据和原始数据(训练集，验证集，测试集)
        self.test_loader = DataLoader(dataset=self.datasets,
                   batch_size=config.batch_size,
                   collate_fn=test_data_loader.collate_fn,
                   shuffle=False,                   
                   num_workers=4,
                   drop_last=False) 
        return self.test_loader,self.ori_data 

    def predict(self,data):
        for index,i in enumerate(data):
            i['sentence'] = list(i['sentence'])
            data[index] = i   
        self.data_loader,self.ori_data = self.dataload(data)
        self.model.eval()
        pred_result = []
        label_result = []
        result = []
        total_ent_r = 0
        total_ent_p = 0
        total_ent_c = 0
        i = 0
        
        with torch.no_grad():
            for data_batch in self.data_loader:
                sentence_batch = self.ori_data[i:i+config.batch_size]
                # entity_text = data_batch[-1]
                data_batch = [data.cuda() for data in data_batch]
                bert_inputs, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

                outputs = self.model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
                length = sent_length

                grid_mask2d = grid_mask2d.clone()

                outputs = torch.argmax(outputs, -1)
                decode_entities = utils.test_decode(outputs.cpu().numpy(), length.cpu().numpy())

                for ent_list, sentence in zip(decode_entities, sentence_batch):
                    sentence = sentence["sentence"]
                    instance = {"sentence": sentence, "entity": []}
                    for ent in ent_list:
                        instance["entity"].append({"text": [sentence[x] for x in ent[0]],
                                                    "type": config.vocab.id_to_label(ent[1])})
                    result.append(instance)

        #         total_ent_r += ent_r
        #         total_ent_p += ent_p
        #         total_ent_c += ent_c

        #         grid_labels = grid_labels[grid_mask2d].contiguous().view(-1)
        #         outputs = outputs[grid_mask2d].contiguous().view(-1)

        #         label_result.append(grid_labels.cpu())
        #         pred_result.append(outputs.cpu())
        #         i += config.batch_size

        # label_result = torch.cat(label_result)
        # pred_result = torch.cat(pred_result)

        return result


if __name__ == "__main__":
    # test_data = [{"sentence":"氧氟沙星滴眼液适用于治疗细菌性结膜炎、角膜炎、角膜溃疡、泪囊炎、术后感染等外眼感染。"}]
    test_data = [{"sentence":"对本品及肾上腺皮质激素类药物有过敏史者禁用，特殊情况下权衡利弊使用，注意病情恶化的可能：高血压、血栓症、胃与十二指肠溃疡、精神病、电解质代谢异常、心肌梗塞、内脏手术、青光眼等患者一般不宜使用"}]
    # test_data = [{"sentence": '常建良,男,'},{"sentence": "历任公司采供处处长, 兼党支部书记和工作处处长。"}]
    # test_data2 = [{"sentence": ["历任公司采供处处长, 兼党支部书记和工作处处长。"]}]
    # result = predict("Final", test_loader, ori_data)
    # print(result)
    singlepredict = SinglePredict(0,'model.pt')
    result = singlepredict.predict(test_data)
    for i in result:
        print(i['entity'])
    # print(singlepredict.predict(test_data2))
    #print(singlepredict.result)
