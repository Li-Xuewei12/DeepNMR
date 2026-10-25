from pathlib import Path
import pandas as pd,numpy as np,json,argparse
from scipy.stats import binomtest
parser=argparse.ArgumentParser(description="Recompute frozen-cohort metrics, bootstrap intervals, and paired tests")
parser.add_argument('--data-root',type=Path,default=Path('data/evaluation'))
parser.add_argument('--output',type=Path,default=Path('runs/frozen_evaluation'))
args=parser.parse_args();D=args.data_root;O=args.output
O.mkdir(parents=True,exist_ok=False)
rng=np.random.default_rng(20261024)
metrics=[];tests=[];checks=[]
def holm(v):
 v=np.array(v);ix=np.argsort(v);out=np.empty(len(v));out[ix]=np.minimum(1,np.maximum.accumulate(v[ix]*(len(v)-np.arange(len(v)))));return out
def compare(arr,den,stage,cohort,nuc,endpoint,models):
 # Equal reference targets within an aligned cohort; swap all errors of a molecule together.
 obs=arr.sum(0)/den.sum();boot=[]
 for start in range(0,5000,50):
  ix=rng.integers(len(den),size=(min(50,5000-start),len(den)));boot.extend(arr[ix].sum(1)/den[ix].sum(1)[:,None])
 lo,hi=np.quantile(boot,[.025,.975],axis=0)
 for j,m in enumerate(models):
  for r in metrics:
   if (r['stage'],r['cohort'],r['nucleus'],r['endpoint'],r['model'])==(stage,cohort,nuc,endpoint,m):r.update(lo=lo[j],hi=hi[j])
 diffs=arr[:,0,None]-arr[:,1:];ext=np.zeros(len(models)-1)
 observed=diffs.sum(0)
 for start in range(0,19999,100):
  signs=(rng.integers(2,size=(min(100,19999-start),len(den)))*2-1).astype(float)
  ext+=(np.abs(signs@diffs)>=np.abs(observed)-1e-9).sum(0)
 for j,m in enumerate(models[1:]):tests.append(dict(stage=stage,cohort=cohort,nucleus=nuc,endpoint=endpoint,model_a=models[0],model_b=m,difference=obs[0]-obs[j+1],p=(ext[j]+1)/20000,n_molecules=len(den)))
def calc(f,y,p,uid):
 e=f[p].to_numpy(float)-f[y].to_numpy(float);return dict(n=len(f),molecules=f[uid].nunique(),MAE=np.abs(e).mean(),RMSE=np.sqrt((e*e).mean()),R2=1-(e*e).sum()/((f[y]-f[y].mean())**2).sum(),P95=np.quantile(abs(e),.95),macro_MAE=f.assign(a=abs(e)).groupby(uid).a.mean().mean())
for co,label in [('Internal_DFT8K_test','DFT8K'),('External_Dataset3_test','Dataset3')]:
 for n in ['1H','13C']:
  arrays=[]; ref=None
  for model in ['DeepNMR','CASCADE','GT-NMR','NMRNet']:
   pp=list((D/'Stage_delta').glob(f'{model}*__{co}__{n}__atom_errors.parquet'));assert len(pp)==1,(model,pp)
   f=pd.read_parquet(pp[0]).sort_values(['molecule_uid','atom_index_conformer']).reset_index(drop=True)
   keys=f[['molecule_uid','atom_index_conformer']]
   if ref is not None:
    pd.testing.assert_frame_equal(keys,ref[0]);np.testing.assert_allclose(f.target_ppm,ref[1],atol=3e-5,rtol=1e-6)
   else:ref=(keys,f.target_ppm.to_numpy())
   metrics.append(dict(stage='delta',cohort=label,nucleus=n,endpoint='Assigned MAE',model=model,**calc(f,'target_ppm','prediction_ppm','molecule_uid')))
   g=f.assign(err=abs(f.prediction_ppm-f.target_ppm)).groupby('molecule_uid').err.agg(['sum','count']);arrays.append(g['sum'].values)
  compare(np.stack(arrays,1),g['count'].values,'delta',label,n,'Assigned MAE',['DeepNMR','CASCADE','GT-NMR','NMRNet'])
  checks.append(dict(cohort=label,nucleus=n,aligned_atoms=len(f),molecules=len(g)))
 print('delta',label,'verified',flush=True)
for co,label in [('Internal_CHAMPS_test','CHAMPS internal'),('External_CHAMPS_holdout','CHAMPS holdout')]:
 models=['DeepNMR','PaiNN','EquiformerV2','Cormorant'];frames=[];ref=None
 for model in models:
  f=pd.read_parquet(D/'Stage_J'/f'{model}__{co}__pair_errors.parquet');f=f.rename(columns={'target_hz':'target_total_hz','prediction_hz':'predicted_j_hz'});cols=['molecule_id','coupling_type'];f=f.sort_values(cols+['target_total_hz']).reset_index(drop=True)
  if ref is not None:
   pd.testing.assert_frame_equal(f[cols],ref[0]);np.testing.assert_allclose(f.target_total_hz,ref[1],atol=5e-5,rtol=1e-6)
  else:ref=(f[cols],f.target_total_hz.to_numpy())
  pred='prediction_total_hz' if 'prediction_total_hz' in f else 'predicted_j_hz'
  f['pred']=f[pred];frames.append(f)
 for typ in ['All']+sorted(frames[0].coupling_type.unique()):
  arrays=[]
  for m,f in zip(models,frames):
   f=f if typ=='All' else f[f.coupling_type==typ]
   metrics.append(dict(stage='J',cohort=label,nucleus=typ,endpoint='Coupling MAE',model=m,**calc(f,'target_total_hz','pred','molecule_id')))
   g=f.assign(err=abs(f.pred-f.target_total_hz)).groupby('molecule_id').err.agg(['sum','count']);arrays.append(g['sum'].values)
  compare(np.stack(arrays,1),g['count'].values,'J',label,typ,'Coupling MAE',models)
 print('J',label,'verified',flush=True)
pd.DataFrame(metrics).to_csv(O/'recomputed_metrics.csv',index=False)
t=pd.DataFrame(tests)
for (stage,cohort),ids in t.groupby(['stage','cohort']).groups.items():t.loc[ids,'p_holm']=holm(t.loc[ids,'p'])
t['symbol']=t.p_holm.apply(lambda p:'***' if p<.001 else '**' if p<.01 else '*' if p<.05 else 'ns');t.to_csv(O/'paired_tests_delta_J.csv',index=False)
# Retain the user's original Figure 5 model and values, explicitly separate lineage.
q=pd.read_csv(D/'Retrieval/current_Fig5_query_hits.csv')
rr=[]
for mod in ['H','C','H+C']:
 f=q[q.modality==mod];base=f[f.model=='joint_core'].set_index('sample_name')
 for m in ['nmrnet','nmrsolver_500g_fbmo']:
  b=f[f.model==m].set_index('sample_name').loc[base.index]
  for k in range(1,6):
   c=f'hit_at_{k}_connectivity';x=base[c].astype(bool);y=b[c].astype(bool);n10=int((x&~y).sum());n01=int((~x&y).sum());rr.append(dict(modality=mod,k=k,model_b=m,n10=n10,n01=n01,p=binomtest(n10,n10+n01,.5).pvalue if n10+n01 else 1))
t=pd.DataFrame(rr)
# Each endpoint/modality is an explicitly distinct exploratory 2-comparator family; Fig5 tests Top-5 H+C only.
for _,ix in t.groupby(['modality','k']).groups.items():t.loc[ix,'p_holm']=holm(t.loc[ix,'p'])
t.to_csv(O/'retrieval_tests_current_Fig5.csv',index=False)
(O/'analysis_checks.json').write_text(json.dumps(checks,indent=2));print('All paired analyses finished',flush=True)
