"""Reproduce final-model analyses from the released anonymous case probabilities."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize,brentq
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve,precision_recall_curve

THRESHOLD=0.3530539465287861
BRANCHES=['clinical_probability','conventional_radiomics_probability','habitat_probability','cnn_2d_probability','cnn_2_5d_probability']
MODELS=BRANCHES+['ohp_probability']
COHORTS={'development_oof':296,'internal_validation':126,'external':156,'prospective':93}

def auc_vectors(y,p):
    pos=np.asarray(p)[y==1];neg=np.asarray(p)[y==0]
    matrix=(pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:])
    return matrix.mean(),matrix.mean(axis=1),matrix.mean(axis=0)

def paired_auc(y,a,b):
    aa,ap,an=auc_vectors(y,a);bb,bp,bn=auc_vectors(y,b)
    cov=np.cov([ap,bp])/len(ap)+np.cov([an,bn])/len(an)
    variance=max(0.,float(cov[0,0]+cov[1,1]-2*cov[0,1]));sd=np.sqrt(variance);delta=float(aa-bb)
    pvalue=float(2*norm.sf(abs(delta/sd))) if sd else (1. if not delta else 0.)
    return {'auc_difference':delta,'ci_lower':delta-1.95996398454*sd,'ci_upper':delta+1.95996398454*sd,'delong_p':pvalue}

def holm(values):
    p=np.asarray(values);order=np.argsort(p);out=np.empty(len(p))
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out

def net_benefit(y,p,t):
    predicted=p>=t
    return float(((predicted&(y==1)).sum()-(predicted&(y==0)).sum()*t/(1-t))/len(y))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(__file__).resolve().parents[1]/'data/predictions/all_cohorts_predictions.csv')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();data=pd.read_csv(args.input,float_precision='round_trip')
    required=['case_id','cohort','ohp_label',*MODELS,'classification_threshold','predicted_ohp_positive']
    if any(c not in data for c in required):raise ValueError('Missing a documented prediction column.')
    if data.case_id.duplicated().any() or not data.case_id.str.fullmatch(r'P[0-9a-f]{20}').all():raise ValueError('Expected unique anonymous case codes.')
    if data.cohort.value_counts().to_dict()!=COHORTS:raise ValueError('Expected the documented 671-case cohort allocation.')
    if not data.ohp_label.isin([0,1]).all():raise ValueError('OHP labels must be binary.')
    if not np.isfinite(data[MODELS]).all().all() or not ((data[MODELS]>=0)&(data[MODELS]<=1)).all().all():raise ValueError('Invalid model probability.')
    if not np.allclose(data[BRANCHES].mean(axis=1),data.ohp_probability,rtol=0,atol=1e-12):raise ValueError('The five-branch mean does not match the supplied final probabilities.')
    if not np.array_equal(data.classification_threshold.to_numpy(),np.full(len(data),THRESHOLD)):raise ValueError('Unexpected classification threshold.')
    if not np.array_equal((data.ohp_probability>=THRESHOLD).astype(int),data.predicted_ohp_positive):raise ValueError('Final classifications differ from the fixed-threshold rule.')
    out=args.output_dir
    if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new or empty output directory.')
    out.mkdir(parents=True,exist_ok=True)
    discrimination=[];primary=[];paired=[];calibration=[];dca=[];roc=[];pr=[];calibration_bins=[]
    for cohort in COHORTS:
        group=data[data.cohort.eq(cohort)];y=group.ohp_label.to_numpy(dtype=int);p=group.ohp_probability.to_numpy()
        for model in MODELS:
            q=group[model].to_numpy()
            discrimination.append({'cohort':cohort,'model':model,'n':len(y),'ohp_positive':int(y.sum()),'auc':roc_auc_score(y,q),'average_precision':average_precision_score(y,q),'brier':brier_score_loss(y,q)})
            fpr,tpr,cut=roc_curve(y,q,drop_intermediate=False)
            roc.extend({'cohort':cohort,'model':model,'false_positive_rate':a,'sensitivity':b,'threshold':c} for a,b,c in zip(fpr,tpr,cut))
            precision,recall,cut=precision_recall_curve(y,q)
            pr.extend({'cohort':cohort,'model':model,'precision':a,'recall':b,'threshold':c} for a,b,c in zip(precision,recall,np.r_[cut,np.nan]))
        v=p>=THRESHOLD;tp=int((v&(y==1)).sum());fn=int((~v&(y==1)).sum());fp=int((v&(y==0)).sum());tn=int((~v&(y==0)).sum())
        primary.append({'cohort':cohort,'n':len(y),'threshold':THRESHOLD,'TP':tp,'FN':fn,'FP':fp,'TN':tn,'accuracy':(tp+tn)/len(y),'sensitivity':tp/(tp+fn),'specificity':tn/(fp+tn),'PPV':tp/(tp+fp) if tp+fp else np.nan,'NPV':tn/(tn+fn) if tn+fn else np.nan,'F1':2*tp/(2*tp+fp+fn),'mean_probability':p.mean(),'observed_proportion':y.mean()})
        lp=logit(np.clip(p,1e-10,1-1e-10));intercept=brentq(lambda b:np.sum(expit(lp+b)-y),-50,50)
        fit=minimize(lambda b:np.sum(np.logaddexp(0,b[0]+b[1]*lp)-y*(b[0]+b[1]*lp)),[0,1],method='BFGS',tol=1e-10)
        calibration.append({'cohort':cohort,'n':len(y),'mean_probability':p.mean(),'observed_proportion':y.mean(),'expected_observed_ratio':p.mean()/y.mean(),'calibration_in_large':intercept,'logistic_intercept':fit.x[0],'calibration_slope':fit.x[1]})
        bins=pd.qcut(pd.Series(p),q=6,duplicates='drop')
        for bin_number,label in enumerate(bins.cat.categories):
            mask=(bins==label).to_numpy();n=int(mask.sum());observed=float(y[mask].mean())
            z=norm.ppf(.975);denominator=1+z*z/n
            centre=(observed+z*z/(2*n))/denominator
            half=z*np.sqrt(observed*(1-observed)/n+z*z/(4*n*n))/denominator
            calibration_bins.append({'cohort':cohort,'bin':bin_number,'n':n,'mean_probability':p[mask].mean(),'observed_proportion':observed,'wilson_low':centre-half,'wilson_high':centre+half})
        for threshold in np.unique(np.r_[np.arange(.05,.601,.01),THRESHOLD]):
            dca.append({'cohort':cohort,'threshold':threshold,'five_branch_mean':net_benefit(y,p,threshold),'clinical_branch':net_benefit(y,group.clinical_probability.to_numpy(),threshold),'review_all':y.mean()-(1-y.mean())*threshold/(1-threshold),'review_none':0.})
        if cohort in ['external','prospective']:
            rows=[dict(cohort=cohort,comparator=model,**paired_auc(y,p,group[model].to_numpy())) for model in BRANCHES]
            for row,adj in zip(rows,holm([r['delong_p'] for r in rows])):row['holm_p']=float(adj)
            paired.extend(rows)
    background=data.loc[data.cohort.eq('development_oof'),BRANCHES].mean()
    contribution=data[data.cohort.ne('development_oof')][['case_id','cohort','ohp_label','ohp_probability']].copy()
    contribution['background_probability']=background.mean()
    for model in BRANCHES:contribution[model.replace('_probability','_contribution')]=(data.loc[contribution.index,model]-background[model])/5
    parts=[x for x in contribution if x.endswith('_contribution')]
    if not np.allclose(contribution.background_probability+contribution[parts].sum(axis=1),contribution.ohp_probability,atol=1e-12,rtol=0):raise ValueError('Probability decomposition mismatch.')
    for name,rows in [('model_discrimination.csv',discrimination),('primary_classification_metrics.csv',primary),('paired_auc_comparisons.csv',paired),('calibration_point_estimates.csv',calibration),('calibration_bins.csv',calibration_bins),('decision_curve_points.csv',dca),('roc_points.csv',roc),('precision_recall_points.csv',pr)]:pd.DataFrame(rows).to_csv(out/name,index=False,float_format='%.17g')
    contribution.to_csv(out/'patient_branch_contributions.csv',index=False,float_format='%.17g')
    metadata={'input_cases':len(data),'cohort_sizes':COHORTS,'classification_threshold':THRESHOLD,'fusion_weights':[.2]*5,'output':'Numerical results and curve coordinates; no plots generated.','paired_auc_adjustment':'Holm within the five final-model versus single-branch comparisons in each evaluation cohort.','calibration':'Diagnostic regression estimates only; input probabilities are not changed.','calibration_bins':6,'calibration_bin_method':'Within-cohort probability quantiles using pandas.qcut; duplicates are dropped.','calibration_bin_intervals':'Wilson 95% confidence intervals','contribution_units':'probability; multiply by 100 for percentage points','contribution_background':'development OOF branch means'}
    (out/'ANALYSIS_PARAMETERS.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print(json.dumps({'completed':True,'cases':len(data),'output_directory':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
