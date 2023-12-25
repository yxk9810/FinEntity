import torch
import logging
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import datetime
import config
from model.bert_crf import BertCrfForNer as BertNER
from torch.utils.data import DataLoader, random_split
from transformers import BertTokenizer
from model.bert_crf import BertCrfForNer
from transformers import get_linear_schedule_with_warmup
from torch import cuda
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from seqeval.metrics import f1_score
from seqeval.metrics import precision_score
from seqeval.metrics import accuracy_score
from seqeval.metrics import recall_score
from seqeval.metrics import classification_report
from util.process import ids_to_labels,Metrics,Metrics_e,SpanEntityScore
from seqeval.scheme import BILOU
from util.adversairal import FGM 
from transformers import BertTokenizerFast
from sequence_aligner.dataset import PredictDatasetCRF,PredictDatasetBySeq
from sequence_aligner.containers import TraingingBatch,PredictBatch
from transformers import BertTokenizerFast
from sequence_aligner.labelset import LabelSet
from torch.utils.data import DataLoader

def train_epoch(e,model, data_loader,optimizer,scheduler,device):
    model.train()
    fgm = FGM(model)
    losses = 0.0
    for step, d in enumerate(data_loader):
        step += 1
        input_ids = d["input_ids"]
        attention_mask = d["attention_masks"].type(torch.uint8)
        targets = d["labels"]
        inputs = {
            'input_ids':input_ids.to(device),
            'attention_mask':attention_mask.to(device),
            'labels':targets.to(device)
        }
        outputs = model(
            **inputs
        )
        loss = outputs[0] 
        losses += loss.item()
        loss.backward()
        
        #fgm
        fgm.attack() 
        loss_adv = model( **inputs)[0]
        loss_adv.backward() 
        fgm.restore() 
    
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print("Epoch: {}, train Loss:{:.4f}".format((e+1), losses/step))
    return losses/step

def valid_epoch(e,model, data_loader,device,label_set):
    model.eval()
    y_true, y_pred = [], []
    losses = 0
    all_step = 0 
    with torch.no_grad():
        for step, d in enumerate(data_loader):
            y_true_sub, y_pred_sub = [], []
            input_ids = d["input_ids"]
            attention_mask = d["attention_masks"].type(torch.uint8)
            targets = d["labels"]
            val_input = {
                'input_ids':input_ids.to(device),
                'attention_mask':attention_mask.to(device),
                'labels':targets.to(device)
            }
            outputs = model(
                **val_input
            )
            tmp_eval_loss, logits = outputs[:2]
            losses+=tmp_eval_loss.item()
            tags = model.crf.decode(logits, d['attention_masks'].to(device))
            tags = tags.squeeze(0).cpu().numpy().tolist()
            out_label_ids = d['labels'].cpu().numpy().tolist()
            for i, label in enumerate(out_label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if m == -1:
                        continue
                    temp_1.append(label_set.ids_to_label[out_label_ids[i][j]])
                    temp_2.append(label_set.ids_to_label[tags[i][j]])
                y_true.append(temp_1)
                y_pred.append(temp_2)
            all_step+=1 
    report=classification_report(y_true, y_pred, mode='strict', scheme=BILOU)
    print(report)
    valid_loss = losses/all_step 
    print("Epoch: {}, train Loss:{:.4f}".format((e+1), valid_loss))
    return report

        
def valid_epoch_not_crf(e,model, val_loader,device,label_set):
    model.eval()
    trues, preds = [], []
    losses = 0
    with torch.no_grad():
        for step, d in enumerate(val_loader):
            sub_preds, sub_trues = [],[]
            input_ids = d["input_ids"]
            attention_mask = d["attention_masks"].type(torch.uint8)
            targets = d["labels"]
            val_input = {
                'input_ids':input_ids.to(device),
                'attention_mask':attention_mask.to(device),
                'labels':targets.to(device)
            }
            outputs = model(
                **val_input
            )
            tmp_eval_loss, logits = outputs[:2]
            
            sub_preds =np.argmax(logits.cpu().numpy(), axis=2).reshape(-1).tolist()
            sub_trues = d["labels"].detach().cpu().numpy().reshape(-1).tolist()
            # data process
            gold_labeled,pred_labeled = ids_to_labels(label_set,sub_trues,sub_preds)
            trues.append(gold_labeled)
            preds.append(pred_labeled)
    report=classification_report(trues, preds, mode='strict', scheme=BILOU)
    print(report)

def predict4entity2sequence(model,data,device,tokenizer,label_set,save_list,nlp):
    if len(data) ==0:
        return
    dataset = PredictDatasetBySeq(data=data, tokenizer=tokenizer, label_set=label_set,tokens_per_batch = 128)
    if len(dataset) ==0:
        return
    pred_loader = DataLoader(dataset, batch_size=16, collate_fn=PredictBatch, shuffle=True)
    result_entity,result_sentiment = predict(model, pred_loader,device,label_set,tokenizer)
    if len(data)>512:
        data = data[:511]
    result = nlp(data)
    sub_save = {}
    sub_save['text'] = data
    sub_save['entity'] = result_entity
    sub_save['ent_sentiment'] = result_sentiment
    sub_save['seq_sentiment'] = result[0]['label']
    save_list.append(sub_save)
    
def predict4news(model,data,device,tokenizer,label_set,save_list):
    dataset = PredictDatasetCRF(data=data, tokenizer=tokenizer, label_set=label_set,tokens_per_batch = 128)
    pred_loader = DataLoader(dataset, batch_size=16, collate_fn=PredictBatch, shuffle=True)
    result_entity,result_sentiment = predict(model, pred_loader,device,label_set,tokenizer)
    sub_save = {}
    sub_save['date'] = data["date_publish"]
    sub_save['content'] = data["content"]
    sub_save['entity'] = result_entity
    sub_save['sentiment'] = result_sentiment
    save_list.append(sub_save)
    
def predict(model, data_loader,device,label_set,tokenizer):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for step, d in enumerate(data_loader):
            y_pred_sub = []
            input_ids = d["input_ids"]
            attention_mask = d["attention_masks"].bool()
            val_input = {
                'input_ids':input_ids.to(device),
                'attention_mask':attention_mask.to(device),
            }
            logits = model(
                **val_input
            )[0]
            tags = model.crf.decode(logits, d['attention_masks'].to(device))
            tags = tags.squeeze(0).cpu().numpy().tolist()
            y_pred = []
            result_entity = []
            result_sentiment = []
            
            for i,tag in enumerate(tags):
                temp = []
                sub_tag = tags[i]
                content = tokenizer.decode([token for token in input_ids[i] if token!=0])
                conten_list = tokenizer.tokenize(content)
                id_to_decode = []
                for j, item in enumerate(sub_tag):
                    if(attention_mask[i][j]==True):
                        temp.append(label_set.ids_to_label[tags[i][j]])
                        
                        t = label_set.ids_to_label[tags[i][j]]
     
                        ind = input_ids[i][j].item()
                        if t != "O":
                            if t.startswith("B"):
                                id_to_decode.append(ind)
                            if t.startswith("I"):
                                id_to_decode.append(ind)
                            if t.startswith("L"):
                                id_to_decode.append(ind)
                                
                        if t!="O" and t.startswith("U"):
                            result_entity.append(tokenizer.decode(ind))
                            result_sentiment.append(t.strip('U-'))
                            
                        if t!="O" and t.startswith("L") and j-1>0 and label_set.ids_to_label[tags[i][j-1]] != "O":
                            tokens = tokenizer.convert_ids_to_tokens([token for token in id_to_decode if token!=0])
                            string = tokenizer.convert_tokens_to_string(tokens)
                            result_entity.append(string)
                            id_to_decode = []
                            result_sentiment.append(t.strip('L-'))
                        
                y_pred.append(temp)
            return result_entity,result_sentiment

def train_epoch_span(e, model, data_loader, optimizer, scheduler, device):
    model.train()
    fgm = FGM(model)
    losses = 0.0
    for step, d in enumerate(data_loader):
        step += 1
        input_ids = d["input_ids"]
        attention_mask = d["attention_masks"].type(torch.uint8)
        start_ids = d["start_ids"]
        end_ids = d["end_ids"]

        inputs = {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'start_ids': start_ids.to(device),
            'end_ids': start_ids.to(device)

        }
        outputs = model(
            **inputs
        )
        loss = outputs[0]
        # print(len(outputs))
        # print(outputs[0].size())
        losses += loss.item()
        loss.backward()

        # fgm
        fgm.attack()
        loss_adv = model(**inputs)[0]
        loss_adv.backward()
        fgm.restore()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print("Epoch: {}, train Loss:{:.4f}".format((e + 1), losses / step))
    return losses / step
def bert_extract_item(start_logits, end_logits):
    S = []
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][:-1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][:-1]
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                S.append((s_l, i, i + j))
                break
    return S
def extract_item(start_ids,end_ids):
    import sys 
    T = []
    for i, s_l in enumerate(start_ids):
        if s_l == 0:
            continue
        for j in range(i,len(end_ids)):
            e_l = end_ids[j]
            if s_l == e_l:
                T.append((s_l, i, j))
                break
    return T
def valid_epoch_span(e,model, val_loader,device,label_set,tokenizer):
    id2label = {0:'O',1:'Positive',2:'Negative',3:'Neutral'}
    metric = SpanEntityScore(id2label)

    model.eval()
    trues, preds = [], []
    losses = 0
    all_steps =0 
    with torch.no_grad():
        for step, d in enumerate(val_loader):
            sub_preds, sub_trues = [],[]
            input_ids = d["input_ids"]
            attention_mask = d["attention_masks"].type(torch.uint8)
            start_ids = d["start_ids"]
            end_ids = d["end_ids"]
            val_input = {
                'input_ids':input_ids.to(device),
                'attention_mask':attention_mask.to(device),
                'start_ids': start_ids.to(device),
                'end_ids': start_ids.to(device)
            }
            outputs = model(
                **val_input
            )
            tmp_eval_loss, start_logits, end_logits = outputs[:3]
            start_logits = torch.argmax(start_logits, -1).cpu().numpy()
            end_logits =   torch.argmax(end_logits, -1).cpu().numpy()
            input_ids = input_ids.cpu().numpy()
            start_ids = start_ids.cpu().numpy()
            end_ids = end_ids.cpu().numpy()
            pred_ents = []
            gold_ents = []
            gold_entities = [] 
            pred_entities = [] 
            for tmp_start_logits, tmp_end_logits,tmp_start_ids,tmp_end_ids,tmp_input_ids in zip(start_logits,end_logits,start_ids,end_ids,input_ids):
                R = extract_item(tmp_start_logits.tolist(),tmp_end_logits.tolist())
                T = extract_item(tmp_start_ids.tolist(),tmp_end_ids.tolist())
                # print(tmp_start_logits.tolist())
                # print(tmp_end_logits.tolist())
                # sys.exit(1)
                pred_ents.extend(R)
                gold_ents.extend(T)
                if T:
                    for t_ids in T:
                        tokens = tmp_input_ids.tolist()[t_ids[1]:t_ids[2]+1]
                        text = tokenizer.decode(tokens)
                        type =  id2label[t_ids[0]]
                        gold_entities.append({'type':type,'text':text})
                if R:
                    for t_ids in R:
                        tokens = tmp_input_ids.tolist()[t_ids[1]:t_ids[2]+1]
                        text = tokenizer.decode([tok_id for tok_id in tokens if tok_id!=0])
                        type =  id2label[t_ids[0]]
                        if text.strip()!='':
                            pred_entities.append({'type':type,'text':text})
                # print(gold_entities)

                golds = set([d['text'] for d in gold_entities])
                preds = set([d['text'] for d in pred_entities])
                rrecall = (float(len(golds&preds))+1e-9)/(len(golds)+1e-9)
                precsion = (float(len(golds&preds))+1e-9)/(len(preds)+1e-9)
                print("recall: {:.2f}, eval Loss:{:.4f}".format(rrecall,precsion))
                # print(pred_entities[:1])
            
    #         R = bert_extract_item(start_logits, end_logits)
    #         T = extract_item(start_ids,end_ids)
    #         print(R[:2])
    #         print(T[:2])
            
    #         # sys.exit(1)
            metric.update(true_subject=gold_ents, pred_subject=pred_ents)
    #         losses+=tmp_eval_loss.item()
    #         # sub_preds =np.argmax(logits.cpu().numpy(), axis=2).reshape(-1).tolist()
    #         # #sub_trues = d["labels"].detach().cpu().numpy().reshape(-1).tolist()
    #         # data process
    #         #gold_labeled,pred_labeled = ids_to_labels(label_set,sub_trues,sub_preds)
    #         #trues.append(gold_labeled)
    #         #preds.append(pred_labeled)
            all_steps+=1
    # # report=classification_report(trues, preds, mode='strict', scheme=BILOU)
    # # print(report)
    # print('train loss'+str(float(losses)/all_steps))
    eval_info, entity_info = metric.result()
    results = {f'{key}': value for key, value in eval_info.items()}
    print("Epoch: {}, eval Loss:{:.4f}".format((e+1), losses/all_steps))
    print(results)


