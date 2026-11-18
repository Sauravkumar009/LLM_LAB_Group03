#!/usr/bin/env python3
"""Assignment 5: From-scratch encoder, causal decoder, sweeps, bonus depth study.

Install: pip install torch numpy matplotlib scikit-learn
Full run: python assignment5_complete.py --ag-train train.csv --ag-test test.csv --shakespeare shakespeare.txt --output results
The Shakespeare input must be the text corpus extracted from the specified Kaggle Shakespeare Plays dataset.
For a dependency/data-free code check: python assignment5_complete.py --smoke-test --output smoke_results
"""
import argparse, csv, json, math, random, re, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

LABELS = ['World', 'Sports', 'Business', 'Sci/Tech']

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def parse_ag(path):
    rows = []
    with open(path, encoding='utf-8-sig', newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3 or not row[0].strip().isdigit(): continue
            label = int(row[0]) - 1
            if label not in range(4): continue
            rows.append((label, (row[1]+' '+row[2]).strip()))
    return rows

def words(text): return re.findall(r'[a-z0-9]+', text.lower())

def balanced_subset(rows, per_class=5000, seed=42):
    rng = random.Random(seed); groups = {i:[] for i in range(4)}
    for label, txt in rows: groups[label].append((label,txt))
    if any(len(v) < per_class for v in groups.values()):
        raise ValueError('AG News must have >=5000 training rows per class for the full assignment')
    selected = sum((rng.sample(groups[i],per_class) for i in range(4)), [])
    rng.shuffle(selected); return selected

def build_vocab(rows, size=15000):
    counts=Counter(tok for _,txt in rows for tok in words(txt))
    vocab={'[PAD]':0,'[UNK]':1,'[CLS]':2}
    for t,_ in counts.most_common(size-3): vocab[t]=len(vocab)
    return vocab

class NewsData(Dataset):
    def __init__(self, rows, vocab, max_len=64):
        self.rows=rows; self.vocab=vocab; self.max_len=max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        label,txt=self.rows[i]
        tokens=[2]+[self.vocab.get(t,1) for t in words(txt)][:self.max_len-1]
        valid=len(tokens)
        return (torch.tensor(tokens+[0]*(self.max_len-valid),dtype=torch.long),
                torch.tensor([True]*valid+[False]*(self.max_len-valid)),
                torch.tensor(label,dtype=torch.long))

def sinusoidal(max_len,d):
    pos=torch.arange(max_len,dtype=torch.float32).unsqueeze(1)
    freq=torch.exp(torch.arange(0,d,2,dtype=torch.float32)*(-math.log(10000.)/d))
    pe=torch.zeros(max_len,d); pe[:,0::2]=torch.sin(pos*freq)
    pe[:,1::2]=torch.cos(pos*freq)
    return pe

class ManualAttention(nn.Module):
    def __init__(self,d,h,causal=False):
        super().__init__(); assert d%h==0
        self.h=h; self.dk=d//h; self.causal=causal
        self.q=nn.Linear(d,d); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d); self.o=nn.Linear(d,d)
        self.capture=False; self.last_weights=None
    def forward(self,x,valid=None):
        b,t,d=x.shape
        def split(proj): return proj(x).reshape(b,t,self.h,self.dk).transpose(1,2)
        q,k,v=split(self.q),split(self.k),split(self.v)
        scores=(q@k.transpose(-2,-1))/math.sqrt(self.dk)
        if self.causal:
            upper=torch.ones(t,t,dtype=torch.bool,device=x.device).triu(1)
            scores=scores.masked_fill(upper,float('-inf'))
        if valid is not None: scores=scores.masked_fill(~valid[:,None,None,:],float('-inf'))
        attn=torch.softmax(scores,dim=-1)
        self.last_weights=attn.detach().cpu() if self.capture else None
        y=(attn@v).transpose(1,2).contiguous().reshape(b,t,d)
        return self.o(y)

class EncoderLayer(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.attn=ManualAttention(d,h)
        self.norm1=nn.LayerNorm(d); self.norm2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.ReLU(),nn.Linear(2*d,d))
        self.drop=nn.Dropout(.1)
    def forward(self,x,mask):
        x=self.norm1(x+self.drop(self.attn(x,mask)))
        return self.norm2(x+self.drop(self.ff(x)))

class NewsEncoder(nn.Module):
    def __init__(self,vocab,d=128,h=4,L=2,pos='learned',seq=64):
        super().__init__(); self.pos_type=pos; self.emb=nn.Embedding(vocab,d,padding_idx=0)
        if pos=='learned': self.pos=nn.Embedding(seq,d)
        else: self.register_buffer('fixed_pos',sinusoidal(seq,d))
        self.blocks=nn.ModuleList([EncoderLayer(d,h) for _ in range(L)])
        self.head=nn.Linear(d,4); self.drop=nn.Dropout(.1)
    def forward(self,x,mask):
        t=x.size(1); p=torch.arange(t,device=x.device)
        out=self.drop(self.emb(x)+(self.pos(p) if self.pos_type=='learned' else self.fixed_pos[:t]))
        for block in self.blocks: out=block(out,mask)
        return self.head(out[:,0,:])

class CharData(Dataset):
    def __init__(self,ids,seq=128,stride=64):
        self.ids=ids; self.seq=seq; self.stride=stride
    def __len__(self): return max(0,1+(len(self.ids)-self.seq-1)//self.stride)
    def __getitem__(self,i):
        off=i*self.stride; return self.ids[off:off+self.seq],self.ids[off+1:off+self.seq+1]

class DecoderLayer(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.n1=nn.LayerNorm(d);self.n2=nn.LayerNorm(d)
        self.attn=ManualAttention(d,h,causal=True)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x):
        x=x+self.attn(self.n1(x)); return x+self.ff(self.n2(x))

class CharDecoder(nn.Module):
    def __init__(self,vocab,d=128,h=4,L=4,pos='learned',seq=128):
        super().__init__();self.pos_type=pos;self.emb=nn.Embedding(vocab,d)
        if pos=='learned': self.pos=nn.Embedding(seq,d)
        else: self.register_buffer('fixed_pos',sinusoidal(seq,d))
        self.blocks=nn.ModuleList([DecoderLayer(d,h) for _ in range(L)])
        self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,vocab)
    def forward(self,x):
        t=x.shape[1]; p=torch.arange(t,device=x.device)
        out=self.emb(x)+(self.pos(p) if self.pos_type=='learned' else self.fixed_pos[:t])
        for b in self.blocks: out=b(out)
        return self.head(self.norm(out))

def train_model(model, train_loader, val_loader, task, device, epochs=10, lr=.001, tol=.0001):
    model.to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    hist={'train_loss':[],'val_loss':[],'seconds':[]}; previous=None
    for epoch in range(epochs):
        start=time.perf_counter();model.train();total=0.;seen=0
        for batch in train_loader:
            batch=[t.to(device) for t in batch]
            if task=='news': x,mask,y=batch;logits=model(x,mask);loss=F.cross_entropy(logits,y)
            else:
                x,y=batch;logits=model(x);loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1))
            optimizer.zero_grad();loss.backward();optimizer.step()
            total+=float(loss.item())*len(x);seen+=len(x)
        avg=total/max(seen,1);hist['train_loss'].append(avg)
        model.eval();total=0.;seen=0
        with torch.no_grad():
            for batch in val_loader:
                batch=[t.to(device) for t in batch]
                if task=='news': x,mask,y=batch;logits=model(x,mask);loss=F.cross_entropy(logits,y)
                else:
                    x,y=batch;logits=model(x);loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),y.reshape(-1))
                total+=float(loss.item())*len(x);seen+=len(x)
        vloss=total/max(seen,1)
        hist['val_loss'].append(vloss);hist['seconds'].append(time.perf_counter()-start)
        print(f'{task} epoch={epoch+1} train={avg:.5f} validation={vloss:.5f} sec={hist["seconds"][-1]:.1f}',flush=True)
        if previous is not None and abs(avg-previous)<tol:
            hist['converged']=True;break
        previous=avg
    hist.setdefault('converged',False)
    return hist

def plot_curves(results, out, task, key, title):
    plt.figure(figsize=(9,5))
    for name,r in results.items():
        values=r['history'][key]
        plt.plot(range(1,len(values)+1),values,marker='o',markersize=2,label=name)
    plt.title(title);plt.xlabel('Epoch');plt.ylabel(key.replace('_',' ').title());plt.legend(fontsize=7)
    plt.grid(alpha=.2);plt.tight_layout();plt.savefig(out/f'{task}_{key}.png',dpi=150);plt.close()

def heatmap(weights,title,path,rows=None):
    plt.figure(figsize=(7,5));plt.imshow(weights,aspect='auto',cmap='viridis',interpolation='nearest');plt.colorbar(label='Attention weight')
    plt.xlabel('Key position');plt.ylabel('Query position');plt.title(title)
    if rows is not None: plt.yticks(range(len(rows)),rows)
    plt.tight_layout();plt.savefig(path,dpi=160);plt.close()

def evaluate_news(model,loader,device):
    true=[];pred=[];model.eval()
    with torch.no_grad():
        for x,mask,y in loader:
            logits=model(x.to(device),mask.to(device));true.extend(y.tolist());pred.extend(logits.argmax(-1).cpu().tolist())
    return dict(test_accuracy=100*accuracy_score(true,pred),macro_f1=100*f1_score(true,pred,average='macro'),confusion_matrix=confusion_matrix(true,pred,labels=[0,1,2,3]).tolist())

def generate(model,prompt,stoi,itos,device,temperature=.8,greedy=False,n=150,seq=128):
    ids=[stoi[c] for c in prompt if c in stoi]
    if not ids: raise ValueError('No known characters in prompt')
    model.eval()
    for _ in range(n):
        x=torch.tensor([ids[-seq:]],device=device)
        with torch.no_grad(): logits=model(x)[0,-1]
        nex=int(logits.argmax()) if greedy else int(torch.multinomial(torch.softmax(logits/temperature,dim=0),1))
        ids.append(nex)
    return ''.join(itos[i] for i in ids)

def run_news(args,device,out,smoke=False):
    if smoke:
        rows=[(i,f'{LABELS[i]} headline {i} sample {j} reporter event news') for i in range(4) for j in range(8)]
        test_rows=rows[-8:];train_rows=rows[:24];vrows=rows[::8]
    else:
        train_rows=balanced_subset(parse_ag(args.ag_train),seed=args.seed)
        test_rows=parse_ag(args.ag_test)
        if len(test_rows)!=7600: print('NOTE: test size is',len(test_rows))
        random.Random(args.seed).shuffle(train_rows)
        vcount=max(4,int(len(train_rows)*.1));vrows=train_rows[:vcount];train_rows=train_rows[vcount:]
    vocab=build_vocab(train_rows)
    train=DataLoader(NewsData(train_rows,vocab),batch_size=args.batch,shuffle=True)
    val=DataLoader(NewsData(vrows,vocab),batch_size=args.batch)
    test=DataLoader(NewsData(test_rows,vocab),batch_size=args.batch)
    configs=[(2,2,64,'learned'),(2,4,128,'learned'),(4,2,128,'learned'),(4,4,128,'learned'),(2,4,128,'sinusoidal'),(4,4,128,'sinusoidal')]
    if smoke: configs=configs[:1]
    results={};best=None;best_score=-1
    for L,h,d,pos in configs:
        name=f'L{L}_h{h}_d{d}_{pos}'; print('NEWS',name,flush=True)
        m=NewsEncoder(len(vocab),d,h,L,pos)
        hist=train_model(m,train,val,'news',device,epochs=1 if smoke else args.epochs,lr=args.lr)
        metrics=evaluate_news(m,val,device)
        record={'config':{'layers':L,'heads':h,'d_model':d,'position':pos},'history':hist,'validation_accuracy':metrics['test_accuracy'],'validation_macro_f1':metrics['macro_f1']}
        results[name]=record
        if record['validation_macro_f1']>best_score: best_score=record['validation_macro_f1'];best=(name,m)
    best_name,model=best
    results[best_name]['held_out_test']=evaluate_news(model,test,device)
    model.blocks[-1].attn.capture=True
    x,mask,_=next(iter(test));model.eval()
    with torch.no_grad(): model(x[:1].to(device),mask[:1].to(device))
    a=model.blocks[-1].attn.last_weights[0,:,0,:].numpy()
    actual=int(mask[0].sum());heatmap(a[:,:min(actual,32)],'CLS bidirectional attention by head',out/'task1_cls_attention.png',rows=[f'Head {i+1}' for i in range(a.shape[0])])
    plot_curves(results,out,'task1','train_loss','Task 1: training loss vs epoch')
    torch.save({'model':model.cpu().state_dict(),'config':results[best_name]['config'],'vocab':vocab},out/'task1_best.pt')
    return results

def run_decoder(args,device,out,smoke=False):
    if smoke:
        raw=('ROMEO: What light through yonder window breaks?\nKING HENRY: My lord, to be or not to be.\n'*15)
    else:
        raw=Path(args.shakespeare).read_text(encoding='utf-8')
    chars=sorted(set(raw));stoi={c:i for i,c in enumerate(chars)}
    ids=torch.tensor([stoi[c] for c in raw],dtype=torch.long)
    seq=16 if smoke else 128;cut=int(len(ids)*.9)
    train=DataLoader(CharData(ids[:cut],seq,seq//2),batch_size=min(args.batch,8) if smoke else args.batch,shuffle=True)
    val=DataLoader(CharData(ids[cut:],seq,seq),batch_size=args.batch)
    configs=[(2,2,64,'learned'),(2,4,128,'learned'),(4,2,64,'learned'),(4,4,128,'learned'),(2,4,128,'sinusoidal'),(4,4,128,'sinusoidal'),(6,4,128,'sinusoidal'),(8,4,128,'sinusoidal')]
    if smoke: configs=configs[:1]
    results={};best=None;score=float('inf')
    for L,h,d,pos in configs:
        name=f'L{L}_h{h}_d{d}_{pos}';print('DECODER',name,flush=True)
        m=CharDecoder(len(chars),d,h,L,pos,seq)
        hist=train_model(m,train,val,'decoder',device,epochs=1 if smoke else args.epochs,lr=args.lr)
        v=hist['val_loss'][-1]
        results[name]={'config':{'layers':L,'heads':h,'d_model':d,'position':pos},'history':hist,'final_validation_ce':v,'final_validation_ppl':math.exp(v),'average_seconds_per_epoch':float(np.mean(hist['seconds'])),'epochs_run':len(hist['train_loss'])}
        if v<score:best=(name,m);score=v
    name,model=best;model.blocks[0].attn.capture=True;model.eval()
    with torch.no_grad(): model(next(iter(val))[0][:1].to(device))
    attn=model.blocks[0].attn.last_weights[0,0,:min(seq,32),:min(seq,32)].numpy()
    heatmap(attn,'Causal triangular attention (head 1)',out/'task2_causal_attention.png')
    prompts=['ROMEO:','To be, or not','KING HENRY:','What light through','My lord,']
    temps=[None,.5,.8,1.2,.8]
    generations=[]
    for prompt,temp in zip(prompts,temps):
        text=generate(model,prompt,stoi,chars,device,greedy=temp is None,temperature=temp or 1.,n=8 if smoke else 150,seq=seq)
        generations.append({'prompt':prompt,'method':'greedy' if temp is None else 'sample','temperature':temp,'output':text})
    plot_curves(results,out,'task2','train_loss','Task 2: training loss vs epoch')
    plot_curves(results,out,'task2','val_loss','Task 2: validation loss vs epoch')
    torch.save({'model':model.cpu().state_dict(),'config':results[name]['config'],'stoi':stoi},out/'task2_best.pt')
    (out/'task2_generations.json').write_text(json.dumps(generations,indent=2),encoding='utf8')
    return results

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ag-train',type=Path);ap.add_argument('--ag-test',type=Path);ap.add_argument('--shakespeare',type=Path)
    ap.add_argument('--output',type=Path,default=Path('assignment5_results'))
    ap.add_argument('--epochs',type=int,default=10);ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--lr',type=float,default=.001);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--smoke-test',action='store_true',help='Uses tiny synthetic data to test all code paths, NOT a valid experiment')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    if not args.smoke_test:
        for name in ['ag_train','ag_test','shakespeare']:
            val=getattr(args,name)
            if val is None or not val.is_file(): ap.error(f'--{name.replace("_","-")} must point to a real file (required for genuine metrics)')
    seed_all(args.seed);torch.set_num_threads(min(4,torch.get_num_threads()))
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device,'smoke:',args.smoke_test,flush=True)
    start=time.perf_counter()
    news=run_news(args,device,args.output,args.smoke_test)
    decoder=run_decoder(args,device,args.output,args.smoke_test)
    result={'status':'synthetic_smoke_test_NOT_assignment_results' if args.smoke_test else 'full_training','news':news,'decoder':decoder,'wall_seconds':time.perf_counter()-start}
    (args.output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Wrote',args.output/'results.json',flush=True)

if __name__=='__main__': main()
