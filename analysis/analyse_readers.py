"""Reproduce the final sixteen-reader analysis without changing source exports."""
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET
import json, hashlib, itertools, math, sys, shutil
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path('reader_exports')
OUT = Path('reader_analysis_output')
REFERENCE = Path('approved_reference.csv')
THRESHOLD = 0.3530539465287861
NBOOT = 4000
SEED = 20261107
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sheet(path, number):
    with ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(e.itertext()) for e in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        rows=[]
        tree=ET.fromstring(z.read(f'xl/worksheets/sheet{number}.xml'))
        for row in tree.findall('.//s:sheetData/s:row',NS):
            vals={}
            for c in row:
                letters=''.join(x for x in c.attrib['r'] if x.isalpha())
                ix=0
                for ch in letters: ix=ix*26+ord(ch)-64
                typ=c.get('t'); v=c.find('s:v',NS)
                if typ=='inlineStr': value=''.join(c.find('s:is',NS).itertext())
                elif v is None: value=None
                elif typ=='s': value=strings[int(v.text)]
                elif typ in ['str','e']: value=v.text
                elif typ=='b': value=v.text=='1'
                else:
                    value=float(v.text)
                    if value.is_integer(): value=int(value)
                vals[ix-1]=value
            rows.append([vals.get(i) for i in range(max(vals,default=-1)+1)])
        width=max(map(len,rows)); rows=[r+[None]*(width-len(r)) for r in rows]
        return pd.DataFrame(rows[1:],columns=rows[0])

def write(df,name,sub='02_results'):
    p=OUT/sub/name;p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False,encoding='utf-8-sig')

def auc_h(prob,y):
    return (prob[:,y==1,None]>prob[:,None,y==0]).astype(float)+.5*(prob[:,y==1,None]==prob[:,None,y==0])

def exact_signed_rank(delta):
    delta=np.asarray(delta);delta=delta[np.abs(delta)>1e-12]
    if not len(delta):return 1.
    ranks=rankdata(np.round(abs(delta),12));obs=abs(np.dot(np.sign(delta),ranks))
    signs=2*((np.arange(2**len(delta))[:,None]>>np.arange(len(delta)))&1)-1
    return float((np.abs(signs@ranks)>=obs-1e-10).mean())

def holm(p):
    p=np.asarray(p);order=np.argsort(p);out=np.zeros(len(p));out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def metric_values(votes,probs,y,time=None,confidence=None):
    # Two conditions x readers x cases. AUC is a mean of within-reader AUCs.
    tp=(votes*y).sum(axis=(1,2));fn=((1-votes)*y).sum(axis=(1,2));fp=(votes*(1-y)).sum(axis=(1,2));tn=((1-votes)*(1-y)).sum(axis=(1,2))
    out={'accuracy':(tp+tn)/(tp+tn+fp+fn),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),
         'PPV':tp/(tp+fp),'NPV':tn/(tn+fn),'F1':2*tp/(2*tp+fp+fn),
         'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),
         'AUC':np.array([np.mean([roc_auc_score(y,p) for p in ps]) for ps in probs]),
         'Brier':((probs-y)**2).mean(axis=(1,2))}
    if time is not None:out['reading_seconds']=time.mean(axis=(1,2))
    if confidence is not None:out['confidence']=confidence.mean(axis=(1,2))
    return out,{'TP':tp,'FN':fn,'FP':fp,'TN':tn}

def bootstrap(votes,probs,y,time,confidence,nboot=NBOOT,seed=SEED):
    nr,nc=votes.shape[1:];ip=np.where(y==1)[0];ineg=np.where(y==0)[0];rng=np.random.default_rng(seed)
    rw=np.random.default_rng(SEED+1000000).multinomial(nr,np.ones(nr)/nr,size=nboot)
    wp=rng.multinomial(len(ip),np.ones(len(ip))/len(ip),size=nboot)
    wn=rng.multinomial(len(ineg),np.ones(len(ineg))/len(ineg),size=nboot)
    cw=np.zeros((nboot,nc));cw[:,ip]=wp;cw[:,ineg]=wn
    # The same reader and case samples are used in both conditions.
    def count(arr):return np.einsum('br,trc,bc->tb',rw,arr,cw,optimize=True)
    tp=count(votes*y);fn=count((1-votes)*y);fp=count(votes*(1-y));tn=count((1-votes)*(1-y))
    out={'accuracy':(tp+tn)/(nr*nc),'sensitivity':tp/(nr*len(ip)),'specificity':tn/(nr*len(ineg)),
         'PPV':tp/(tp+fp),'NPV':tn/(tn+fn),'F1':2*tp/(2*tp+fp+fn),
         'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),
         'Brier':count((probs-y)**2)/(nr*nc),'reading_seconds':count(time)/(nr*nc),'confidence':count(confidence)/(nr*nc)}
    aa=[]
    for ps in probs:
        h=auc_h(ps,y)
        per_reader=np.einsum('bi,rij,bj->br',wp,h,wn,optimize=True)/(len(ip)*len(ineg))
        aa.append((per_reader*rw).sum(axis=1)/nr)
    out['AUC']=np.array(aa)
    return out,(rw,cw,wp,wn)

def summarize(votes,probs,y,time,confidence,label,seed):
    point,counts=metric_values(votes,probs,y,time,confidence)
    boot,weights=bootstrap(votes,probs,y,time,confidence,seed=seed)
    reader_points=[metric_values(votes[:,i:i+1],probs[:,i:i+1],y,time[:,i:i+1],confidence[:,i:i+1])[0] for i in range(votes.shape[1])]
    rows=[]
    for key,p in point.items():
        ci=np.quantile(boot[key],[.025,.975],axis=1);dboot=boot[key][1]-boot[key][0];dc=np.quantile(dboot,[.025,.975])
        pv=exact_signed_rank([r[key][1]-r[key][0] for r in reader_points])
        rows.append(dict(group=label,readers=votes.shape[1],cases=len(y),positive=int(y.sum()),metric=key,
                         without_AI=p[0],without_low=ci[0,0],without_high=ci[1,0],with_AI=p[1],with_low=ci[0,1],with_high=ci[1,1],
                         delta=p[1]-p[0],delta_low=dc[0],delta_high=dc[1],reader_signed_rank_p=pv,holm_p=np.nan))
    family=['accuracy','sensitivity','specificity','AUC'];adj=holm([r['reader_signed_rank_p'] for r in rows if r['metric'] in family])
    j=0
    for r in rows:
        if r['metric'] in family:r['holm_p']=adj[j];j+=1
    return rows,boot,weights

def main():
    for sub in ['01_inputs','02_results','03_scripts','04_figures']:(OUT/sub).mkdir(parents=True,exist_ok=True)
    records=[];audit=[];events=[];times=[];hashes=[];fail=[]
    for f in sorted(ROOT.glob('R??_\u539f\u59cb\u5bfc\u51fa.xlsx')):
        rid=f.name[:3];raw=sheet(f,2);ev=sheet(f,3);info=dict(sheet(f,1).values)
        assert len(raw)==214 and raw.reader_id.eq(rid).all()
        assert raw.groupby('phase').size().to_dict()=={'ai_assisted':107,'unassisted':107}
        assert not raw.duplicated(['reader_id','phase','case_id']).any()
        assert not raw.record_id.duplicated().any() and not ev.event_id.duplicated().any()
        for c,lo,hi in [('human_probability_pct',0,100),('binary_judgment',0,1),('confidence_1_5',1,5)]:
            assert raw[c].between(lo,hi).all() and raw[c].mod(1).eq(0).all()
        assert raw.ai_exposure_before_unassisted.astype(str).str.lower().eq('false').all()
        assert raw.loc[raw.phase=='unassisted',['ai_probability','ai_label','ai_threshold']].isna().all().all()
        raw['case_started_at']=pd.to_datetime(raw.case_started_at,utc=True);raw['submitted_at']=pd.to_datetime(raw.submitted_at,utc=True)
        elapsed=(raw.submitted_at-raw.case_started_at).dt.total_seconds()*1000
        assert (raw.active_reading_ms>0).all() and (raw.active_reading_ms<=elapsed+1).all()
        assert abs(raw.reading_time_min-raw.active_reading_ms/60000).max()<=.000051
        for typ,col in [('case_started','case_started_at'),('submit','submitted_at')]:
            en=ev[ev.event_type==typ].set_index('record_id');rn=raw.set_index('record_id');assert set(en.index)==set(rn.index)
            q=en.join(rn,lsuffix='_event',rsuffix='_row');assert (q.case_id_event==q.case_id_row).all()
            assert (pd.to_datetime(q['at'],utc=True)==pd.to_datetime(q[col],utc=True)).all()
        u=raw[raw.phase=='unassisted'];a=raw[raw.phase=='ai_assisted'];gap=(a.case_started_at.min()-u.submitted_at.max()).total_seconds()/86400
        assert set(u.case_id)==set(a.case_id)
        for ph,g in raw.groupby('phase'):
            assert sorted(g.phase_sequence_index)==list(range(1,108))
            z=g.sort_values('case_started_at');assert not (z.case_started_at.iloc[1:].reset_index(drop=True)<z.submitted_at.iloc[:-1].reset_index(drop=True)).any()
        times.append(dict(reader_id=rid,configured_days=info.get('washout_days_configured'),actual_interval_days=gap,
                          unaided_start=u.case_started_at.min(),unaided_end=u.submitted_at.max(),assisted_start=a.case_started_at.min(),assisted_end=a.submitted_at.max()))
        hashes.append(dict(file=f.name,sha256=sha(f),bytes=f.stat().st_size));records.append(raw);events.append(ev)
        audit.append(dict(reader_id=rid,records=len(raw),unaided=107,assisted=107,case_pairing='pass',record_event_times='pass',range_checks='pass',time_checks='pass'))
    assert len(records)==16
    d=pd.concat(records,ignore_index=True);assert not d.record_id.duplicated().any()
    group=pd.read_csv(ROOT/'\u533b\u751f\u5206\u7ec4.csv',encoding='gb18030').rename(columns={'\u533b\u751f\u7f16\u53f7':'reader_id','\u79d1\u5ba4':'specialty','\u5e74\u8d44':'seniority'})
    group.specialty=group.specialty.map({'\u4e34\u5e8a\u79d1\u5ba4':'Urology','\u5f71\u50cf\u79d1\u5ba4':'Radiology'});group.seniority=group.seniority.map({'\u4e2d\u7ea7':'Mid-career','\u9ad8\u5e74\u8d44':'Senior'})
    assert group.specialty.value_counts().to_dict()=={'Urology':9,'Radiology':7}
    d=d.merge(group,on='reader_id',validate='many_to_one')
    snapshot=OUT/'01_inputs/case_reference_107.csv'
    if snapshot.exists():ref=pd.read_csv(snapshot,dtype={'ID':str})
    else:
        ext=pd.read_csv(REFERENCE,dtype={'ID':str});ref=ext[ext.cohort.eq('external')&ext.ID.isin(d.case_id.unique())].copy();assert len(ref)==107
        write(ref,'case_reference_107.csv','01_inputs')
    assert not ref.ID.duplicated().any() and set(ref.ID)==set(d.case_id)
    d=d.merge(ref[['ID','y','size','age','sex','ohp_probability']],left_on='case_id',right_on='ID',validate='many_to_one')
    assert d.y.notna().all() and d['size'].notna().all()
    a=d[d.phase=='ai_assisted'];assert np.allclose(a.ai_probability,a.ohp_probability,rtol=0,atol=1e-12)
    assert (a.ai_threshold==THRESHOLD).all() and (a.ai_label==(a.ai_probability>=THRESHOLD).astype(int)).all()
    assert a.groupby('case_id').ai_probability.nunique().eq(1).all()
    assert ((d.human_probability_pct>=50).astype(int)==d.binary_judgment).all()
    cases=sorted(ref.ID);readers=sorted(group.reader_id);phases=['unassisted','ai_assisted']
    y=ref.set_index('ID').loc[cases,'y'].to_numpy(int);size=ref.set_index('ID').loc[cases,'size'].to_numpy(float)
    def arr(col):return np.stack([d[d.phase.eq(ph)].pivot(index='reader_id',columns='case_id',values=col).loc[readers,cases].to_numpy(float) for ph in phases])
    votes=arr('binary_judgment');probs=arr('human_probability_pct')/100;time=arr('active_reading_ms')/1000;conf=arr('confidence_1_5')
    rows,boot,weights=summarize(votes,probs,y,time,conf,'All readers',SEED)
    write(pd.DataFrame(rows),'overall_paired_metrics.csv')
    individual=[]
    for i,r in enumerate(readers):
        val,ct=metric_values(votes[:,i:i+1],probs[:,i:i+1],y,time[:,i:i+1],conf[:,i:i+1])
        for k,ph in enumerate(phases):individual.append(dict(reader_id=r,condition=ph,**group.set_index('reader_id').loc[r].to_dict(),**{key:float(v[k]) for key,v in val.items()},**{key:int(v[k]) for key,v in ct.items()}))
    write(pd.DataFrame(individual),'individual_reader_metrics.csv')
    gain_tests=[];idf=pd.DataFrame(individual)
    uu=idf[idf.condition=='unassisted'].set_index('reader_id').loc[readers];aa=idf[idf.condition=='ai_assisted'].set_index('reader_id').loc[readers]
    for dim,first in [('specialty','Urology'),('seniority','Senior')]:
        labels=group.set_index('reader_id').loc[readers,dim].to_numpy();nfirst=int((labels==first).sum())
        assignments=list(itertools.combinations(range(16),nfirst));weights_perm=np.zeros((len(assignments),16))
        for jj,ix in enumerate(assignments):weights_perm[jj,list(ix)]=1/nfirst
        weights_perm-=np.where(weights_perm==0,1/(16-nfirst),0)
        for key in ['accuracy','sensitivity','specificity','AUC']:
            delta=(aa[key]-uu[key]).to_numpy();obs=delta[labels==first].mean()-delta[labels!=first].mean();pv=float((abs(weights_perm@delta)>=abs(obs)-1e-12).mean())
            gain_tests.append(dict(grouping=dim,first_group=first,metric=key,difference_in_gain=obs,permutation_p=pv))
    write(pd.DataFrame(gain_tests),'between_group_gain_tests.csv')
    group_rows=[];group_boot={}
    for dim in ['specialty','seniority']:
        for j,(label,g) in enumerate(group.groupby(dim)):
            ix=[readers.index(r) for r in g.reader_id];rr,bb,_=summarize(votes[:,ix],probs[:,ix],y,time[:,ix],conf[:,ix],f'{dim}: {label}',SEED+100+j)
            group_rows.extend(rr);group_boot[(dim,label)]=bb
    write(pd.DataFrame(group_rows),'specialty_and_seniority.csv')
    # Independent case strata; preserve shared reader weights for the interaction.
    size_rows=[];size_boot={}
    for j,(name,mask) in enumerate([('<=4 cm',size<=4),('>4 cm',size>4)]):
        rr,bb,_=summarize(votes[:,:,mask],probs[:,:,mask],y[mask],time[:,:,mask],conf[:,:,mask],name,SEED+300+j)
        size_rows.extend(rr);size_boot[name]=bb
    write(pd.DataFrame(size_rows),'tumour_size_subgroups.csv')
    # Reuse the same sampled reader multiplicities across size strata, with independent case draws.
    inter=[]
    for key in ['accuracy','sensitivity','specificity','AUC','Brier']:
        s=next(r for r in size_rows if r['group']=='<=4 cm' and r['metric']==key);l=next(r for r in size_rows if r['group']=='>4 cm' and r['metric']==key)
        x=(size_boot['<=4 cm'][key][1]-size_boot['<=4 cm'][key][0])-(size_boot['>4 cm'][key][1]-size_boot['>4 cm'][key][0]);ci=np.quantile(x,[.025,.975])
        inter.append(dict(metric=key,small_minus_large_gain=s['delta']-l['delta'],ci_low=ci[0],ci_high=ci[1]))
    write(pd.DataFrame(inter),'size_gain_interactions.csv')
    # Current, identical AI scores form a case-level comparator, not 1712 independent AI predictions.
    mp=ref.set_index('ID').loc[cases,'ohp_probability'].to_numpy(float);mv=(mp>=THRESHOLD).astype(float)
    mvals,mcounts=metric_values(mv[None,None,:].repeat(2,axis=0),mp[None,None,:].repeat(2,axis=0),y)
    rw,cw,wp,wn=weights;mc={}
    mtp=cw@(mv*y);mfn=cw@((1-mv)*y);mfp=cw@(mv*(1-y));mtn=cw@((1-mv)*(1-y))
    mc.update(accuracy=(mtp+mtn)/len(y),sensitivity=mtp/y.sum(),specificity=mtn/(len(y)-y.sum()),PPV=mtp/(mtp+mfp),NPV=mtn/(mtn+mfn),F1=2*mtp/(2*mtp+mfp+mfn),balanced_accuracy=.5*(mtp/y.sum()+mtn/(len(y)-y.sum())),Brier=cw@((mp-y)**2)/len(y))
    mc['AUC']=np.einsum('bi,ij,bj->b',wp,auc_h(mp[None,:],y)[0],wn,optimize=True)/(sum(y)*sum(1-y))
    model=[]
    for key,p in mvals.items():
        ci=np.quantile(mc[key],[.025,.975]);diff=boot[key][1]-mc[key];dc=np.quantile(diff,[.025,.975]);point=next(r for r in rows if r['metric']==key)
        model.append(dict(metric=key,AI_alone=p[0],AI_low=ci[0],AI_high=ci[1],assisted_readers=point['with_AI'],difference=point['with_AI']-p[0],difference_low=dc[0],difference_high=dc[1]))
    write(pd.DataFrame(model),'AI_alone_and_assisted_readers.csv')
    # Paired error transitions, including both correction and newly introduced errors.
    correct=votes==y;trans=[]
    for name,mask in [('All cases',np.ones(len(y),bool)),('OHP-positive',y==1),('OHP-negative',y==0),('AI correct',mv==y),('AI incorrect',mv!=y)]:
        cu=correct[0,:,mask].T;ca=correct[1,:,mask].T
        for desc,mat in [('Wrong to correct',~cu&ca),('Correct to wrong',cu&~ca),('Correct to correct',cu&ca),('Wrong to wrong',~cu&~ca)]:trans.append(dict(group=name,cases=int(mask.sum()),reader_case_decisions=int(mask.sum()*16),transition=desc,count=int(mat.sum())))
    write(pd.DataFrame(trans),'paired_error_transitions.csv')
    # Agreement and confidence are descriptive supporting analyses.
    agree=[]
    for k,ph in enumerate(phases):
        n1=votes[k].sum(axis=0);pi=(n1*(n1-1)+(16-n1)*(15-n1))/(16*15);pr=n1.mean()/16;kap=(pi.mean()-(pr**2+(1-pr)**2))/(1-(pr**2+(1-pr)**2))
        agree.append(dict(condition=ph,fleiss_kappa=kap,unanimous_cases=int(((n1==0)|(n1==16)).sum()),cases=len(y)))
    write(pd.DataFrame(agree),'interreader_agreement.csv')
    case_rows=[]
    for i,c in enumerate(cases):
        rr=dict(case_id=c,y=int(y[i]),diameter_cm=size[i],AI_probability=mp[i],AI_class=int(mv[i]))
        for k,ph in enumerate(phases):rr.update({ph+'_positive_votes':int(votes[k,:,i].sum()),ph+'_mean_probability':float(probs[k,:,i].mean()),ph+'_correct_votes':int(correct[k,:,i].sum())})
        rr['net_correct_gain']=rr['ai_assisted_correct_votes']-rr['unassisted_correct_votes'];case_rows.append(rr)
    write(pd.DataFrame(case_rows),'case_votes_and_risks.csv')
    descriptive=[]
    for (r,ph),g in d.groupby(['reader_id','phase']):
        for correct_group,h in g.groupby(g.binary_judgment==g.y):
            descriptive.append(dict(reader_id=r,condition=ph,correct=bool(correct_group),n=len(h),mean_confidence=h.confidence_1_5.mean(),mean_reading_seconds=h.active_reading_ms.mean()/1000,median_reading_seconds=h.active_reading_ms.median()/1000))
    write(pd.DataFrame(descriptive),'confidence_and_time_by_correctness.csv')
    # Vote/probability checks address potential row-order linkage errors.
    correlations=[]
    case_df=pd.DataFrame(case_rows)
    for ph in phases:
        for response in [ph+'_positive_votes',ph+'_mean_probability']:
            for reference in ['diameter_cm','AI_probability']:
                rho,pv=spearmanr(case_df[response],case_df[reference]);correlations.append(dict(condition=ph,response=response,reference=reference,spearman_rho=rho,p=pv))
    write(pd.DataFrame(correlations),'case_linkage_correlations.csv')
    write(d,'combined_reading_records_3424.csv','01_inputs');write(group,'reader_groups.csv','01_inputs')
    write(pd.DataFrame(times),'session_intervals.csv');write(pd.DataFrame(audit),'input_integrity_checks.csv')
    hashes.append(dict(file='\u533b\u751f\u5206\u7ec4.csv',sha256=sha(ROOT/'\u533b\u751f\u5206\u7ec4.csv'),bytes=(ROOT/'\u533b\u751f\u5206\u7ec4.csv').stat().st_size));write(pd.DataFrame(hashes),'input_file_SHA256.csv','01_inputs')
    source_meta={'reference_sha256':sha(REFERENCE) if REFERENCE.exists() else None,'reference_snapshot_sha256':sha(snapshot),'threshold':THRESHOLD,'bootstrap_replicates':NBOOT,'seed':SEED,'conditions':phases,'n_readers':16,'n_cases':107,'positive_cases':31,'n_records':len(d),'source_files_unchanged':all(sha(ROOT/r['file'])==r['sha256'] for r in hashes),'interval_min':min(r['actual_interval_days'] for r in times),'interval_max':max(r['actual_interval_days'] for r in times),'interval_median':float(np.median([r['actual_interval_days'] for r in times])),'AI_probability_max_difference':float(abs(a.ai_probability-a.ohp_probability).max()),'reader_binary_probability_50_concordance':1.0}
    (OUT/'analysis_parameters.json').write_text(json.dumps(source_meta,ensure_ascii=False,indent=2),encoding='utf-8')
    def clean(x):
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        if isinstance(x,(float,np.floating)) and not np.isfinite(x):return None
        return x
    (OUT/'analysis_summary.json').write_text(json.dumps(clean({'overall':rows,'groups':group_rows,'size':size_rows,'size_interaction':inter,'model_comparison':model,'transitions':trans,'agreement':agree}),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    dest=OUT/'03_scripts'/Path(__file__).name
    if Path(__file__).resolve()!=dest.resolve():shutil.copy2(__file__,dest)
    print(json.dumps({'overall':rows,'size':[r for r in size_rows if r['metric'] in ['AUC','sensitivity']],'model':[r for r in model if r['metric']=='AUC'],'meta':source_meta},ensure_ascii=False,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True,type=Path,help='Approved R01–R16 original XLSX exports and reader grouping CSV')
    parser.add_argument('--reference',required=True,type=Path,help='Approved final model reference CSV')
    parser.add_argument('--output-dir',required=True,type=Path,help='New analysis output directory')
    args=parser.parse_args()
    ROOT=args.input_dir.resolve();REFERENCE=args.reference.resolve();OUT=args.output_dir.resolve()
    if OUT.exists():raise FileExistsError('Choose a new output directory; frozen outputs are not overwritten')
    main()
