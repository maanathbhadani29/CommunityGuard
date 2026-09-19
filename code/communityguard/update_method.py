"""Targeted response tools. Clean test truth is not an input to any method here.

Llama chooses among real operations. Every operation changes at most one channel
of one building and is checked against an explicit reference with stated trust assumptions.
The interval guarantee is conditional on reference accuracy; calibration is not
claimed to establish exchangeability under seasonal shift.
"""
from pathlib import Path
import json, random
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from .pipeline import build_model
from .study import save_json

CHANNELS = ['electricity_pricing','non_shiftable_load','solar_generation','hour']
LABELS = ['Price','Load','PV','Clock']
OPERATIONS = ['PEER_CONSENSUS','CONDITIONAL_MODEL','AE_RECONSTRUCTION','ENERGY_BALANCE']

def distance(a,b,k):
    d=np.asarray(a)-np.asarray(b)
    return (d+12)%24-12 if k==3 else d

def circular_hour(value):
    return (np.rint(value)-1)%24+1

class RepairTools:
    def __init__(self,names,buildings,config):
        self.names=list(names);self.B=buildings;self.M=len(names);self.config=config
        self.cols=[names.index(c) for c in CHANNELS]
    def shape(self,x):
        return np.asarray(x).reshape(-1,self.B,self.M)
    def channels(self,x):
        return self.shape(x)[:,:,self.cols]
    def context(self,x,b,k):
        a=self.shape(x); peers=[i for i in range(self.B) if i!=b]
        same=a[:,peers,self.cols[k]]/self.scales[peers,k]
        ctx=[]
        for name in ['outdoor_dry_bulb_temperature','outdoor_relative_humidity','diffuse_solar_irradiance','direct_solar_irradiance']:
            ctx.append(np.median(a[:,peers,self.names.index(name)],axis=1))
        h=np.median(a[:,peers,self.names.index('hour')],axis=1)
        mo=np.median(a[:,peers,self.names.index('month')],axis=1)
        day=np.median(a[:,peers,self.names.index('day_type')],axis=1)
        ctx += [np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),np.sin(2*np.pi*(mo-1)/12),np.cos(2*np.pi*(mo-1)/12)]
        ctx += [(day==i).astype(float) for i in range(1,9)]
        return np.column_stack([same,*ctx])
    def peer(self,x,b,k):
        z=self.channels(x);peers=[i for i in range(self.B) if i!=b]
        if k==3:return circular_hour(np.median(z[:,peers,k],axis=1))
        return np.maximum(0,np.median(z[:,peers,k]/self.scales[peers,k],axis=1)*self.scales[b,k])
    def ae_features(self,x):
        z=self.channels(x)
        z=z/self.scales[None,:,:]
        h=self.channels(x)[:,:,3]
        return np.concatenate([z[:,:,:3],np.sin(2*np.pi*h/24)[:,:,None],np.cos(2*np.pi*h/24)[:,:,None]],axis=2).reshape(len(x),-1).astype(np.float32)
    def ae_channels(self,x):
        if getattr(self,"_ae_input",None) is x: return self._ae_value
        z=self.ae_predict(self.ae_features(x)).reshape(len(x),self.B,5)
        out=np.empty((len(x),self.B,4))
        out[:,:,:3]=np.maximum(0,z[:,:,:3]*self.scales[None,:,:3])
        angle=np.arctan2(z[:,:,3],z[:,:,4])*24/(2*np.pi)
        out[:,:,3]=circular_hour(angle)
        self._ae_input=x;self._ae_value=out
        return out
    def fit(self,train,cal,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        z=self.channels(train)
        self.scales=np.maximum(np.quantile(np.abs(z),0.95,axis=0),0.01);self.scales[:,3]=24
        self.models={};self.validators={};self.context_scalers={}
        self.pv_peers={};self.pv_slopes={}
        # Three train-selected, independently scaled peers prevent one unusually
        # small PV installation from dominating a multi-regressor fit.
        for b in range(self.B):
            y=z[:,b,2].astype(float); ranked=[]
            for j in range(self.B):
                if j==b:continue
                v=z[:,j,2].astype(float);energy=float(v@v)
                if energy<=1e-8:continue
                slope=max(0,float(v@y)/energy)
                mse=float(np.mean((slope*v-y)**2))
                ranked.append((mse,j,slope))
            chosen=sorted(ranked)[:3]
            if len(chosen)<3:raise ValueError('At least three nonzero PV peers are required.')
            self.pv_peers[b]=[r[1] for r in chosen]
            self.pv_slopes[b]=np.array([r[2] for r in chosen])
        for b in range(self.B):
            for k in range(3):
                X=self.context(train,b,k); scaler=StandardScaler().fit(X)
                X=scaler.transform(X); y=z[:,b,k]
                self.context_scalers[b,k]=scaler
                self.models[b,k]=Ridge(alpha=self.config['ridge_alpha'],solver='svd').fit(X,y)
                self.validators[b,k]=ExtraTreesRegressor(n_estimators=64,max_depth=12,min_samples_leaf=8,random_state=42,n_jobs=1).fit(X,y)
        torch.manual_seed(self.config['ae_seed']);np.random.seed(self.config['ae_seed']);random.seed(self.config['ae_seed'])
        torch.use_deterministic_algorithms(True)
        self.ae_model,self.ae_predict,self.ae_losses,_=build_model(self.ae_features(train),self.config['ae_epochs'],self.config['ae_learning_rate'],self.config['threads'])
        torch.save(self.ae_model.state_dict(),folder/'feature_aware_ae.pt')
        q=self.references(cal)
        delta=np.empty((self.B,4));truth=self.channels(cal)
        for b in range(self.B):
            for k in range(4):
                err=np.abs(distance(truth[:,b,k],q[:,b,k],k))
                floor=self.config['scale_floor_fraction']*self.scales[b,k] if k<3 else 0.01
                delta[b,k]=max(float(np.quantile(err,self.config['interval_quantile'],method='higher')),floor)
        self.delta=delta
        np.savez_compressed(folder/'calibration.npz',scales=self.scales,delta=delta,calibration_references=q,calibration_channels=truth,ae_losses=self.ae_losses)
        # Trusted serialized fits: load only the supplied package's own files.
        import joblib
        joblib.dump({'models':self.models,'validators':self.validators,'context_scalers':self.context_scalers,'pv_peers':self.pv_peers,'pv_slopes':self.pv_slopes},folder/'predictors.joblib')
        save_json(folder/'training_manifest.json',{'train_hours':len(train),'calibration_hours':len(cal),'scale_source':'training only','interval_source':'clean calibration only','target_forecasts_used':False,'net_consumption_identity_used':True,'pv_reference':'median of three peers selected by training squared error after nonnegative scalar transfer fit; no intercept; training only','calendar':'cyclic covariates and peer clock','validation_model':self.config.get('reference_mode','statistical'),'guarantee':'conditional; calibration quantile is not a distribution-shift guarantee'})
        return self
    def references(self,x):
        if getattr(self,"_reference_input",None) is x: return self._reference_value
        out=np.empty((len(x),self.B,4))
        for b in range(self.B):
            for k in range(4):
                if self.config.get('reference_mode')=='protected_balance' and k in (1,2):
                    pv=np.maximum(0,np.median(self.channels(x)[:,self.pv_peers[b],2]*self.pv_slopes[b][None,:],axis=1))
                    out[:,b,k]=pv if k==2 else np.maximum(0,self.shape(x)[:,b,self.names.index('net_electricity_consumption')]+pv)
                elif k==3 or k==0:
                    # Shared price/clock channels have a separately declared peer-majority reference.
                    out[:,b,k]=self.peer(x,b,k)
                else:
                    X=self.context_scalers[b,k].transform(self.context(x,b,k))
                    out[:,b,k]=np.maximum(0,self.validators[b,k].predict(X))
        self._reference_input=x;self._reference_value=out
        return out
    def candidate(self,x,b,k,operation):
        if operation=='ENERGY_BALANCE':
            a=self.shape(x);net=a[:,b,self.names.index('net_electricity_consumption')]
            if k==1:return np.maximum(0,net+a[:,b,self.cols[2]])
            if k==2:return np.maximum(0,a[:,b,self.cols[1]]-net)
            raise ValueError('ENERGY_BALANCE is offered only for load or PV')
        if operation=='PEER_CONSENSUS':return self.peer(x,b,k)
        if operation=='CONDITIONAL_MODEL':
            if k==3:return self.peer(x,b,k)
            X=self.context_scalers[b,k].transform(self.context(x,b,k))
            return np.maximum(0,self.models[b,k].predict(X))
        if operation=='AE_RECONSTRUCTION':return self.ae_channels(x)[:,b,k]
        raise ValueError('Unrecognized operation')
    def evidence(self,x):
        z=self.channels(x);q=self.references(x);ae=self.ae_channels(x);entries=[]
        for b in range(self.B):
            for k in range(4):
                err=np.abs(distance(z[:,b,k],q[:,b,k],k));delta=self.delta[b,k]
                inconsistent=err>self.config['gate_factor']*delta
                balance_bad=np.ones(len(x),dtype=bool)
                if self.config.get('reference_mode')=='protected_balance' and k in (1,2):
                    a=self.shape(x);net=a[:,b,self.names.index('net_electricity_consumption')]
                    balance_bad=np.abs(net-z[:,b,1]+z[:,b,2])>self.config['balance_tolerance']
                    inconsistent &= balance_bad
                fraction=float(np.mean(inconsistent))
                score=float(np.mean(np.maximum(err/(2*delta)-1,0)*balance_bad))
                entries.append({'building':b+1,'channel':CHANNELS[k],'score':score,'inconsistent_fraction':fraction,'interval_radius':float(delta),'observed_mean':float(z[:,b,k].mean()),'reference_mean':float(q[:,b,k].mean()),'interval_width_normalized':float(delta/self.scales[b,k]),'ae_normalized_rmse':float(np.sqrt(np.mean((distance(z[:,b,k],ae[:,b,k],k)/self.scales[b,k])**2)))})
        entries=sorted(entries,key=lambda v:v['score'],reverse=True)
        alarm=any(e['inconsistent_fraction']>=self.config['record_alarm_fraction'] and e['score']>0 for e in entries)
        return {'alarm':alarm,'candidates':entries[:5],'assumptions':self.config['trusted_assumptions'],'alarm_fraction':self.config['record_alarm_fraction'],'evidence_note':'All evidence derives from received telemetry, declared protected references, and train/calibration fits. True attacked building and clean test values are unavailable.'}
    def verification(self,x,b,k,operation):
        """Return candidate plus entrywise guard; does not accept any clean truth."""
        z=self.channels(x);q=self.references(x)[:,b,k];c=self.candidate(x,b,k,operation)
        delta=self.delta[b,k]
        lower=np.abs(distance(z[:,b,k],q,k))-delta
        upper=np.abs(distance(c,q,k))+delta
        mask=lower>upper+1e-9
        if k==3:valid=np.isfinite(c)&(c>=1)&(c<=24)&(c==np.rint(c))
        else:valid=np.isfinite(c)&(c>=0)
        mask &= valid
        if self.config.get('reference_mode')=='protected_balance' and k in (1,2):
            a=self.shape(x);net=a[:,b,self.names.index('net_electricity_consumption')]
            load=c if k==1 else z[:,b,1]
            pv=c if k==2 else z[:,b,2]
            mask &= np.abs(net-load+pv)<=self.config['balance_tolerance']
        y=np.array(x,copy=True);col=b*self.M+self.cols[k]
        y[mask,col]=c[mask]
        return y,{'accepted':bool(mask.any()),'changed_hours':int(np.sum(y[:,col]!=x[:,col])),'eligible_hours':int(mask.sum()),'mean_guaranteed_absolute_improvement_conditional':float(np.maximum(lower-upper,0)[mask].mean()) if mask.any() else 0.0,'reason':'conditional_interval_dominance' if mask.any() else 'no_entry_passes_interval_dominance','scope':'One building/channel only; guarantee requires true channel within the reference interval.'},mask
    def proposal_options(self,x,building,channel):
        b=building-1;k=CHANNELS.index(channel);out=[]
        for op in OPERATIONS:
            if op=='ENERGY_BALANCE' and (k not in (1,2) or self.config.get('reference_mode')!='protected_balance'): continue
            _,report,_=self.verification(x,b,k,op)
            out.append({'operation':op,**report})
        return out
    def ae_localization(self,x):
        z=self.channels(x);q=self.ae_channels(x)
        errors=np.stack([distance(z[:,:,k],q[:,:,k],k)/self.scales[None,:,k] for k in range(4)],axis=2)**2
        b,k=np.unravel_index(np.argmax(errors.mean(0)),(self.B,4))
        return int(b),int(k),q
