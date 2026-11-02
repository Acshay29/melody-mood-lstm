"""Project 10: LSTM melody generator with controlled temperature comparisons.
A recurrent neural language model, NOT a GAN. Default dataset is procedurally generated
symbolic-note teaching data. Do not present it as real-world music training data.
"""
import argparse, csv, json, math, random, sys, wave
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw

PITCHES = list(range(48, 85))
BOS, EOS = len(PITCHES), len(PITCHES) + 1
VOCAB = len(PITCHES) + 2

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def toy_melodies(count=1000, seed=42, length=20):
    rng=np.random.default_rng(seed)
    scales = [[0,2,4,7,9], [0,3,5,7,10], [0,2,5,7,9]]
    out=[]
    for _ in range(count):
        tonic=int(rng.choice([48, 50, 53, 55, 57, 60]))
        scale=rng.choice(len(scales)); steps=scales[scale]
        pitch_pool=[p for p in [tonic + k + 12*octave for octave in [0,1,2] for k in steps] if p in PITCHES]
        idx=int(rng.integers(0,len(pitch_pool)))
        notes=[]
        for j in range(length):
            idx=int(np.clip(idx+int(rng.choice([-2,-1,0,1,2],p=[.1,.27,.26,.27,.1])),0,len(pitch_pool)-1))
            notes.append(pitch_pool[idx]-PITCHES[0])
        out.append([BOS]+notes+[EOS])
    return out

def read_melodies(path):
    """CSV with melody_id,step,midi; sequences grouped by melody_id."""
    entries={}
    with open(path,newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):
            midi=int(r['midi'])
            if midi not in PITCHES: raise ValueError(f'MIDI {midi} outside supported 48..84')
            entries.setdefault(r['melody_id'],[]).append((int(r['step']),midi-48))
    return [[BOS]+[p for _,p in sorted(items)]+[EOS] for items in entries.values() if len(items)>2]

class MelodyLSTM(nn.Module):
    def __init__(self,embedding=32,hidden=128,layers=2):
        super().__init__()
        self.embedding=nn.Embedding(VOCAB,embedding)
        self.rnn=nn.LSTM(embedding,hidden,num_layers=layers,batch_first=True)
        self.head=nn.Linear(hidden,VOCAB)
        self.config=dict(embedding=embedding,hidden=hidden,layers=layers)
    def forward(self,ids,state=None):
        x=self.embedding(ids)
        x,state=self.rnn(x,state)
        return self.head(x),state

def train(args):
    set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    seqs=read_melodies(args.dataset) if args.dataset else toy_melodies(args.examples,args.seed)
    if not seqs: raise ValueError('No valid melodies found')
    model=MelodyLSTM(args.embedding,args.hidden,args.layers).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    criterion=nn.CrossEntropyLoss(ignore_index=-100)
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    history=[]
    for epoch in range(1,args.epochs+1):
        random.shuffle(seqs); total=0; n_batches=0
        for off in range(0,len(seqs),args.batch_size):
            batch=seqs[off:off+args.batch_size]
            maximum=max(len(s) for s in batch)
            x=torch.full((len(batch),maximum-1),BOS,dtype=torch.long)
            y=torch.full((len(batch),maximum-1),-100,dtype=torch.long)
            for i,s in enumerate(batch):
                x[i,:len(s)-1]=torch.tensor(s[:-1]);y[i,:len(s)-1]=torch.tensor(s[1:])
            x=x.to(device);y=y.to(device)
            logits,_=model(x)
            loss=criterion(logits.reshape(-1,VOCAB),y.reshape(-1))
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
            total+=float(loss.item());n_batches+=1
        avg=total/n_batches
        history.append((epoch,avg,math.exp(min(avg,20))))
        print(f'epoch {epoch}/{args.epochs}: cross_entropy={avg:.4f}, perplexity={math.exp(min(avg,20)):.2f}')
    with open(output/'training_history.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['epoch','cross_entropy','perplexity']);w.writerows(history)
    torch.save({'state_dict':model.state_dict(),'config':model.config,'seed':args.seed,'data_source':str(args.dataset or 'procedural-teaching-sequences')},output/'melody_lstm.pt')
    return output/'melody_lstm.pt'

@torch.inference_mode()
def sample(model,temperature=1.0,seed=7,prefix=(60,62,64,67),length=16,device='cpu'):
    if temperature<=0: raise ValueError('temperature must be > 0')
    model.eval(); model.to(device)
    rng=torch.Generator(device=device).manual_seed(seed)
    state=None
    input_ids=[BOS]+[int(p)-48 for p in prefix]
    for token in input_ids[:-1]:
        _,state=model(torch.tensor([[token]],device=device),state)
    current=input_ids[-1];notes=list(prefix)
    for _ in range(length-len(prefix)):
        logits,state=model(torch.tensor([[current]],device=device),state)
        # Disallow BOS/EOS for fixed-length audible melody.
        probs=torch.softmax(logits[0,-1,:len(PITCHES)]/temperature,dim=0)
        nxt=int(torch.multinomial(probs,1,generator=rng).item())
        current=nxt; notes.append(nxt+48)
    return notes

def notes_to_wav(notes,path,duration=5.0,sr=22050):
    """Sine plus second harmonic, with attack/release; illustrative synthesizer, not neural audio."""
    total=int(duration*sr);audio=np.zeros(total,dtype=np.float64)
    boundaries=np.linspace(0,total,len(notes)+1,dtype=int)
    for i,midi in enumerate(notes):
        a,b=boundaries[i],boundaries[i+1];n=b-a
        t=np.arange(n)/sr; freq=440*2**((midi-69)/12)
        tone=np.sin(2*np.pi*freq*t)+0.18*np.sin(4*np.pi*freq*t)
        attack=np.clip(t/.018,0,1)
        release=np.clip((n/sr-t)/.065,0,1)
        audio[a:b]=tone*attack*release
    audio=.7*audio/(max(np.max(np.abs(audio)),1e-8))
    pcm=(audio*32767).astype('<i2')
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(sr);f.writeframes(pcm.tobytes())

def plot_comparison(results,path,seconds=5.0):
    W,H=1300,720
    im=Image.new('RGB',(W,H),'#fbfcfe');d=ImageDraw.Draw(im)
    d.text((50,28),'THE MELODY THAT CHANGES ITS MOOD',fill='#1b3851')
    palette=['#378a83','#a97ad1','#db786b']
    for j,(label,notes) in enumerate(results.items()):
        top=115+j*192; d.text((55,top-27),f'{label}  |  {len(notes)} notes, {seconds:g} seconds',fill='#1b3851')
        for i,p in enumerate(notes):
            step=1110/max(1,len(notes)); x=135+i*step; y=top+140-(p-48)*3.3
            d.rounded_rectangle((x,y-8,x+max(4,step*.8),y+8),radius=4,fill=palette[j%len(palette)])  # centre bar on its pitch line
        for k in [48,60,72,84]:
            y=top+140-(k-48)*3.3
            d.line((126,y,1245,y),fill='#e8ecf1',width=1)
            d.text((61,y-7),str(k),fill='#6a7986')
    d.text((55,H-28),'Same trained LSTM + same prompt; only temperature changes. The WAV synthesizer is rule-based.',fill='#576d7e')
    im.save(path)

def generate(args):
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=MelodyLSTM(**checkpoint['config']);model.load_state_dict(checkpoint['state_dict'])
    temps=[float(x) for x in args.temperatures.split(',')]
    result={}
    for temp in temps:
        notes=sample(model,temperature=temp,seed=args.seed,prefix=(60,62,64,67),length=args.length)
        label=f'T={temp:g}';result[label]=notes
        slug=f'temperature_{temp:g}'.replace('.','p')
        notes_to_wav(notes,out/f'{slug}.wav',duration=args.seconds)
    with open(out/'melody_comparison.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['temperature','step','midi','frequency_hz'])
        for label,notes in result.items():
            for step,midi in enumerate(notes): w.writerow([label[2:],step,midi,round(440*2**((midi-69)/12),2)])
    plot_comparison(result,out/'melody_comparison.png',args.seconds)
    (out/'run_metadata.json').write_text(json.dumps(dict(model='LSTM next-token predictor (not GAN)',data_source=checkpoint.get('data_source'),temperatures=temps,seed=args.seed,prefix=[60,62,64,67],length=args.length,seconds=args.seconds,provenance='Generated symbolic sequence + deterministic wave synthesis'),indent=2))
    print('Saved comparisons to',out)

def main():
    # Accept the course-wide style `--mode train ...` as well as the original `train ...`.
    argv=sys.argv[1:]
    if '--mode' in argv:
        i=argv.index('--mode'); argv=[argv[i+1]]+argv[:i]+argv[i+2:]
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train');tr.add_argument('--dataset',default=None);tr.add_argument('--examples',type=int,default=400);tr.add_argument('--epochs',type=int,default=12);tr.add_argument('--batch-size',type=int,default=32);tr.add_argument('--embedding',type=int,default=32);tr.add_argument('--hidden',type=int,default=128);tr.add_argument('--layers',type=int,default=2);tr.add_argument('--lr',type=float,default=.002);tr.add_argument('--seed',type=int,default=42);tr.add_argument('--cpu',action='store_true');tr.add_argument('--output-dir',default='outputs/training')
    ge=sub.add_parser('generate');ge.add_argument('--checkpoint',required=True);ge.add_argument('--temperatures',default='0.5,1.0,1.5');ge.add_argument('--seed',type=int,default=7);ge.add_argument('--length',type=int,default=16);ge.add_argument('--seconds',type=float,default=5);ge.add_argument('--output-dir',default='outputs/comparison')
    sm=sub.add_parser('smoke-test');sm.add_argument('--output-dir',default='outputs/smoke_test')
    args=ap.parse_args(argv)
    if args.command=='train':train(args)
    elif args.command=='generate':generate(args)
    else:
        # Tiny run that checks every stage; NOT a trained melody model.
        out=Path(args.output_dir)
        t=ap.parse_args(['train','--epochs','1','--examples','32','--cpu','--output-dir',str(out/'training')])
        ck=train(t)
        generate(ap.parse_args(['generate','--checkpoint',str(ck),'--output-dir',str(out/'comparison')]))
        print('SMOKE TEST PASSED (1 epoch; not a meaningful model):',out)
if __name__=='__main__':main()
