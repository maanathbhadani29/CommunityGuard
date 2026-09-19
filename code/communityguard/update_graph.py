"""Actual conditional LangGraph. Ground truth is excluded from its state."""
from typing import TypedDict,Any
from pathlib import Path
import numpy as np
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import MemorySaver
from .update_method import CHANNELS,OPERATIONS
from .study import save_json

class DefenseState(TypedDict,total=False):
    case_id:str
    x:Any
    evidence:dict
    diagnosis:dict
    proposal:dict
    candidate:Any
    output:Any
    verification:dict
    attempts:int
    inspections:int
    rejected:list
    trace:list
    terminal:str
    error:str

def build_defense_graph(tools,client,folder,max_attempts=2,allow_inspection=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    def event(s,node,**kwargs):return s.get('trace',[])+[{'node':node,**kwargs}]
    def log(s,stage):return folder/s['case_id']/f'{stage}_{s.get("attempts",0)}_{s.get("inspections",0)}.json'
    def detect(s):
        e=tools.evidence(s['x'])
        return {'evidence':e,'attempts':0,'inspections':0,'rejected':[],'error':'','trace':event(s,'detection',alarm=e['alarm'])}
    def forensic(s):
        rec=client.decide('forensics',s['evidence'],log(s,'forensics'))
        d=rec.get('parsed');err=rec.get('error')
        if d and d['building']!=0:
            valid={(r['building'],r['channel']) for r in s['evidence']['candidates']}
            if (d['building'],d['channel']) not in valid:err='Diagnosis outside supplied candidate coordinates';d=None
        return {'diagnosis':d or {},'error':err or '', 'trace':event(s,'forensic_agent',source=client.source,decision=d,error=err)}
    def route_forensic(s):
        if s['error'] or s['diagnosis'].get('building',0)==0:return 'abstain'
        if s['diagnosis'].get('tool_request')=='PEER_SUMMARY' and allow_inspection and s['inspections']<1:return 'inspect'
        return 'plan' if s['diagnosis'].get('assessment')=='corruption_suspected' else 'abstain'
    def inspect(s):
        d=s['diagnosis'];k=CHANNELS.index(d['channel']);v=tools.channels(s['x'])[:,:,k]
        e=dict(s['evidence']);e['peer_summary']=[{'building':b+1,'mean':float(v[:,b].mean()),'p10':float(np.quantile(v[:,b],.1)),'p90':float(np.quantile(v[:,b],.9))} for b in range(tools.B)]
        e['inspection_budget_remaining']=0
        return {'evidence':e,'inspections':s['inspections']+1,'trace':event(s,'peer_inspection',channel=d['channel'])}
    def plan(s):
        d=s['diagnosis'];options=tools.proposal_options(s['x'],d['building'],d['channel'])
        evidence={'diagnosis':d,'operations':options,'rejected_operations':s['rejected'],'verification_feedback':s.get('verification',{}),'attempt':s['attempts']+1,'budget':max_attempts}
        rec=client.decide('planning',evidence,log(s,'planning'))
        p=rec.get('parsed') or {};err=rec.get('error') or ''
        if p.get('operation') in s['rejected']:err='Repeated rejected operation'
        if p.get('operation') not in [r['operation'] for r in options]+['ABSTAIN']:err=err or 'Operation was not offered for this channel'
        return {'proposal':p,'attempts':s['attempts']+1,'error':err,'trace':event(s,'defense_agent',source=client.source,proposal=p,error=err)}
    def verify(s):
        d=s['diagnosis'];p=s['proposal'];y,r,_=tools.verification(s['x'],d['building']-1,CHANNELS.index(d['channel']),p['operation'])
        rejected=list(s['rejected'])
        if not r['accepted']:rejected.append(p['operation'])
        return {'candidate':y,'verification':r,'rejected':rejected,'trace':event(s,'verification',**r)}
    def route_verify(s):
        if s['verification']['accepted']:return 'commit'
        return 'plan' if s['attempts']<max_attempts else 'abstain'
    def commit(s):return {'output':s['candidate'],'terminal':'committed','trace':event(s,'commit')}
    def abstain(s):return {'output':np.array(s['x'],copy=True),'terminal':'abstained','trace':event(s,'abstain',reason=s.get('error') or s.get('verification',{}).get('reason','no supported response'))}
    g=StateGraph(DefenseState)
    for name,fn in [('detect',detect),('forensic',forensic),('inspect',inspect),('plan',plan),('verify',verify),('commit',commit),('abstain',abstain)]:g.add_node(name,fn)
    g.add_edge(START,'detect')
    g.add_conditional_edges('detect',lambda s:'forensic' if s['evidence']['alarm'] else 'abstain',{'forensic':'forensic','abstain':'abstain'})
    g.add_conditional_edges('forensic',route_forensic,{'inspect':'inspect','plan':'plan','abstain':'abstain'})
    g.add_edge('inspect','forensic')
    g.add_conditional_edges('plan',lambda s:'abstain' if s['error'] or s['proposal'].get('operation')=='ABSTAIN' else 'verify',{'abstain':'abstain','verify':'verify'})
    g.add_conditional_edges('verify',route_verify,{'commit':'commit','plan':'plan','abstain':'abstain'})
    g.add_edge('commit',END);g.add_edge('abstain',END)
    return g.compile(checkpointer=MemorySaver())

def run_graph(graph,x,case_id,folder):
    result=graph.invoke({'x':x,'case_id':case_id,'trace':[]},config={'configurable':{'thread_id':case_id},'recursion_limit':30})
    folder=Path(folder)/case_id;folder.mkdir(parents=True,exist_ok=True)
    save_json(folder/'graph_trace.json',{k:result[k] for k in ['case_id','trace','terminal','diagnosis','proposal','verification','attempts','inspections','error'] if k in result})
    np.savez_compressed(folder/'response.npz',input=x,output=result['output'])
    return result
