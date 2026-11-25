"""CPU-only controlled DPO experiment with a tiny autoregressive language model."""
import argparse, csv, json, random, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(12, 24)
        self.rnn = nn.GRU(24, 32, batch_first=True)
        self.head = nn.Linear(32, 12)
    def forward(self, x):
        return self.head(self.rnn(self.emb(x))[0])

def logp(model, seq):
    # Two response tokens; prompt tokens are excluded from the objective.
    logits = model(seq[:, :-1]).log_softmax(-1)
    return logits[:, -2:].gather(-1, seq[:, -2:, None]).squeeze(-1).sum(-1)

def dataset():
    # Held-out prompt combinations, not held-out individual tokens.
    pairs = [(a,b) for a in range(4) for b in range(4,8)]
    random.Random(42).shuffle(pairs)
    def build(items):
        w = torch.tensor([[a,b,8,10] for a,b in items])
        l = torch.tensor([[a,b,9,11] for a,b in items])
        return w,l
    return build(pairs[:12]), build(pairs[12:])

def dpo_loss(w,l,rw,rl,beta):
    return -F.logsigmoid(beta*((w-l)-(rw-rl))).mean()

@torch.no_grad()
def evaluate(model, ref, data, beta):
    w,l=data
    a,b=logp(model,w),logp(model,l)
    delta=(a-b)-(logp(ref,w)-logp(ref,l))
    return {'accuracy':float((a>b).float().mean()), 'margin':float((a-b).mean()), 'relative_margin':float(delta.mean()), 'dpo_loss':float(-F.logsigmoid(beta*delta).mean())}

def run(seed, epochs, beta):
    torch.manual_seed(seed)
    train,test=dataset()
    ref=TinyLM()
    opt=torch.optim.Adam(ref.parameters(),lr=.01)
    # Neutral SFT: both answer styles appear equally often.
    for _ in range(25):
        opt.zero_grad(); loss=-logp(ref,torch.cat(train)).mean(); loss.backward(); opt.step()
    ref.eval(); ref.requires_grad_(False)
    rows=[]
    for method in ('DPO','chosen-SFT'):
        model=TinyLM(); model.load_state_dict(ref.state_dict())
        opt=torch.optim.Adam(model.parameters(),lr=.003)
        w,l=train
        with torch.no_grad(): rw,rl=logp(ref,w),logp(ref,l)
        for epoch in range(epochs+1):
            rows.append(dict(seed=seed,method=method,epoch=epoch,**evaluate(model,ref,test,beta)))
            if epoch==epochs: break
            opt.zero_grad(); a,b=logp(model,w),logp(model,l)
            loss=dpo_loss(a,b,rw,rl,beta) if method=='DPO' else -a.mean()
            loss.backward(); opt.step()
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument('--epochs',type=int,default=30); p.add_argument('--beta',type=float,default=.1); p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44]); p.add_argument('--out',type=Path,default=Path('results')); args=p.parse_args()
    torch.set_num_threads(1); start=time.time(); args.out.mkdir(parents=True,exist_ok=True)
    rows=[r for s in args.seeds for r in run(s,args.epochs,args.beta)]
    with (args.out/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary={}
    for method in ('DPO','chosen-SFT'):
        summary[method]={}
        for ep in (0,args.epochs):
            subset=[r for r in rows if r['method']==method and r['epoch']==ep]
            summary[method][str(ep)]={k:sum(r[k] for r in subset)/len(subset) for k in ('accuracy','margin','relative_margin','dpo_loss')}
    payload=dict(epochs=args.epochs,beta=args.beta,seeds=args.seeds,torch=torch.__version__,seconds=round(time.time()-start,2),summary=summary)
    (args.out/'summary.json').write_text(json.dumps(payload,indent=2)); print(json.dumps(payload,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,title in zip(axes,['accuracy','relative_margin','dpo_loss'],['Held-out preference accuracy','Reference-relative log margin','Held-out DPO loss']):
        for method in ('DPO','chosen-SFT'):
            vals=torch.tensor([[r[key] for r in rows if r['method']==method and r['seed']==s] for s in args.seeds]); mean=vals.mean(0); std=vals.std(0,unbiased=False); x=range(args.epochs+1)
            ax.plot(x,mean,label=method); ax.fill_between(x,mean-std,mean+std,alpha=.15)
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Controlled DPO reproduction | synthetic token pairs | mean ± SD'); fig.tight_layout(); fig.savefig(args.out/'experiment.png',dpi=180); plt.close(fig)
if __name__=='__main__': main()
