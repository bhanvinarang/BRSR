import sys, re, json
sys.path.insert(0,'src')
from tune_retrieval_offline import *
ann,qv,co=load()
pgtext={}
for l in open('data/corpus/pages.jsonl',encoding='utf-8'):
    r=json.loads(l); pgtext[(r['company_id'],r['page'])]=r['text']

T2={
"F01":"Total energy consumption (A+B+C+D+E+F) Joules or multiples Gigajoules from renewable sources non-renewable sources electricity fuel consumption other sources",
"F02":"Total electricity consumption (A) from renewable sources non-renewable sources Joules or multiples Gigajoules",
"F03":"Total Scope 1 emissions Break-up of the GHG into CO2 CH4 N2O HFCs PFCs SF6 NF3 Metric tonnes of CO2 equivalent",
"F04":"Total Scope 2 emissions Break-up of the GHG into CO2 CH4 N2O HFCs PFCs SF6 NF3 Metric tonnes of CO2 equivalent",
"F05":"Scope 3 emissions Metric tonnes of CO2 equivalent value chain Total Scope 3",
"F06":"Total volume of water withdrawal in kilolitres surface water groundwater third party water seawater desalinated water (i + ii + iii + iv + v)",
"F07":"Total volume of water discharged in kilolitres surface water groundwater seawater third-parties",
"F08":"Total waste generated in metric tonnes plastic waste e-waste bio-medical waste construction demolition battery radioactive hazardous non-hazardous waste (A+B + C + D + E + F + G + H)",
"F09":"hazardous waste other hazardous waste generated metric tonnes total waste generated",
"F10":"Employees and workers including differently abled permanent employees other than permanent employees Total employees (D + E) male female",
"F11":"Employees and workers including differently abled permanent employees Female number and percentage Total employees (D + E)",
"F12":"Whether the entity has in place a mechanism to receive and redress grievances Employees and workers grievance redressal policy",
"F13":"Energy intensity per rupee of turnover Total energy consumption Revenue from operations intensity per rupee of turnover adjusted for Purchasing Power Parity intensity in terms of physical output optional the relevant metric may be selected by the entity",
"F14":"Water intensity per rupee of turnover Total water consumption Revenue from operations intensity per rupee of turnover adjusted for Purchasing Power Parity intensity in terms of physical output optional the relevant metric may be selected by the entity",
"F15":"Waste intensity per rupee of turnover Total waste generated Revenue from operations intensity per rupee of turnover adjusted for Purchasing Power Parity intensity in terms of physical output optional the relevant metric may be selected by the entity",
"F16":"Total Scope 1 and Scope 2 emissions per rupee of turnover Total Scope 1 and Scope 2 emission Revenue from operations intensity optional the relevant metric may be selected by the entity",
}
def norm(s): return re.sub(r'[^0-9.]','',s.replace(',',''))
def has_val(text,gv):
    t=re.sub(r'(?<=\d),(?=\d)','',text)
    return re.search(r'(?<![\d.])'+re.escape(gv)+r'(?![\d])',t) is not None
vmask=[(r.status=='answered' and r.field_id!='F12' and r.page.isdigit() and (len(norm(r.gold_value))>=4 or '.' in norm(r.gold_value))) for r in ann.itertuples(index=False)]

def rank_chunks(c,dense,lex,k,depth=50,rrf_k=60,wd=1,wl=1):
    d_pos=np.argsort(-dense,kind='stable')[:depth]; l_pos=np.argsort(-lex,kind='stable')[:depth]
    dr={int(p):r for r,p in enumerate(d_pos,1)}; lr={int(p):r for r,p in enumerate(l_pos,1)}
    cand=set(dr)|set(lr)
    sc={p:wd/(rrf_k+dr.get(p,10**4))+wl/(rrf_k+lr.get(p,10**4)) for p in cand}
    out,seen=[],set()
    for p in sorted(cand,key=lambda x:(-sc[x],x)):
        pg=int(c.pages[p])
        if pg in seen: continue
        seen.add(pg); out.append(p)
        if len(out)>=k: break
    return out

def go(lexf,k=5,neighbor=0,**kw):
    res=[]
    for i,r in enumerate(ann.itertuples(index=False)):
        c=co[r.company_id]
        dense=c.vecs@qv[i]
        lex=bm25(c,lexf(r))
        rk={x:kw[x] for x in('depth','rrf_k','wd','wl') if x in kw}
        res.append(rank_chunks(c,dense,lex,k,**rk))
    return res
def metrics(res,ks=(3,5)):
    out={}
    for k in ks:
        pr=[];pn=[];vr=[];vp=[];ctx=[]
        for i,r in enumerate(ann.itertuples(index=False)):
            c=co[r.company_id]; top=res[i][:k]; pages=[int(c.pages[p]) for p in top]
            if r.page.isdigit():
                hit=int(r.page) in pages
                if r.status in('answered','zero'): pr.append(hit)
                elif r.status=='not_disclosed' and r.label_type!='parts_only': pn.append(hit)
            if vmask[i]:
                gv=norm(r.gold_value)
                vr.append(any(has_val(c.chunks[p]['text'],gv) for p in top))
                vp.append(any(has_val(pgtext[(r.company_id,pg)],gv) for pg in pages))
            ctx.append(sum(len(c.chunks[p]['text']) for p in top))
        out[k]=(np.mean(pr),np.mean(pn),np.mean(vr),np.mean(vp),np.mean(ctx),len(vr))
    return out
def show(tag,res):
    m=metrics(res)
    s=' | '.join(f"k={k}: pg-ans {v[0]:.3f} pg-nd {v[1]:.3f} VAL-chunk {v[2]:.3f} val-page {v[3]:.3f} ctx {v[4]:.0f}" for k,v in m.items())
    print(f"{tag:34s} {s}",flush=True)

base=lambda r: tokenize(f"{r.question} BRSR disclosure table {FIELD_TERMS[r.field_id]}")
t2=lambda r: tokenize(T2[r.field_id])
t2q=lambda r: tokenize(f"{r.question} ")+tokenize(T2[r.field_id])
both=lambda r: base(r)+tokenize(T2[r.field_id])
termsonly=lambda r: tokenize(FIELD_TERMS[r.field_id])
print('numeric value rows:',sum(vmask))
show('baseline hybrid',go(base))
show('baseline wl=1.5',go(base,wl=1.5))
show('terms-only',go(termsonly))
show('T2 template phrases only',go(t2))
show('question+T2',go(t2q))
show('base+T2',go(both))
show('base+T2 wl=1.5',go(both,wl=1.5))
show('T2 only wl=1.5',go(t2,wl=1.5))
