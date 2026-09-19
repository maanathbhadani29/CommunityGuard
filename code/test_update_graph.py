"""Integration tests use scripted agents, never present them as Llama responses."""
from pathlib import Path
import tempfile,unittest,json
import numpy as np
from communityguard.update_graph import build_defense_graph,run_graph,DefenseState
from communityguard.update_agents import strict_parse
from communityguard.study import save_json

class ControlledTools:
    B=2
    def __init__(self,alarm=True):self.alarm=alarm
    def evidence(self,x):return {'alarm':self.alarm,'candidates':[{'building':1,'channel':'electricity_pricing','score':3.,'inconsistent_fraction':1.}],'alarm_fraction':.1}
    def channels(self,x):return np.zeros((len(x),2,4))
    def proposal_options(self,*args):return [{'operation':'PEER_CONSENSUS','accepted':True},{'operation':'AE_RECONSTRUCTION','accepted':False}]
    def verification(self,x,b,k,op):
        y=x.copy();ok=op=='PEER_CONSENSUS'
        if ok:y[:,0]=2
        return y,{'accepted':ok,'reason':'fixture_pass' if ok else 'fixture_rejection'},np.ones(len(x),bool)

class ScriptedClient:
    source='scripted_integration_fixture_not_llama'
    def __init__(self,ops=('PEER_CONSENSUS',),malformed=False,inspect=False):self.ops=list(ops);self.malformed=malformed;self.inspect=inspect;self.n=0
    def decide(self,stage,evidence,path):
        if self.malformed:return {'parsed':None,'error':'fixture invalid JSON'}
        if stage=='forensics':p={'building':1,'channel':'electricity_pricing','assessment':'corruption_suspected','tool_request':'PEER_SUMMARY' if self.inspect and 'peer_summary' not in evidence else 'NONE','justification':'Scripted routing test.'}
        else:p={'operation':self.ops[min(self.n,len(self.ops)-1)],'justification':'Scripted routing test.'};self.n+=1
        return {'parsed':p,'error':None}

class GraphTests(unittest.TestCase):
    def run_case(self,client,alarm=True):
        with tempfile.TemporaryDirectory() as d:
            x=np.ones((3,4));g=build_defense_graph(ControlledTools(alarm),client,Path(d)/'logs',2)
            return x,run_graph(g,x,'fixture',Path(d)/'states')
    def test_clean_abstention_is_identity(self):
        x,r=self.run_case(ScriptedClient(),False);np.testing.assert_array_equal(r['output'],x);self.assertEqual(r['attempts'],0)
    def test_malformed_response_abstains(self):
        x,r=self.run_case(ScriptedClient(malformed=True));np.testing.assert_array_equal(r['output'],x)
    def test_unknown_operation_abstains(self):
        x,r=self.run_case(ScriptedClient(('NO_OVERRIDE',)));np.testing.assert_array_equal(r['output'],x)
    def test_rejected_candidate_replans_then_commits(self):
        x,r=self.run_case(ScriptedClient(('AE_RECONSTRUCTION','PEER_CONSENSUS')))
        self.assertEqual(r['terminal'],'committed');self.assertEqual(r['attempts'],2)
        np.testing.assert_array_equal(r['output'][:,1:],x[:,1:])
        self.assertEqual([e['node'] for e in r['trace']].count('verification'),2)
    def test_unoffered_operation_abstains(self):
        x,r=self.run_case(ScriptedClient(('ENERGY_BALANCE',)));np.testing.assert_array_equal(r['output'],x)
    def test_repeated_rejection_rolls_back(self):
        x,r=self.run_case(ScriptedClient(('AE_RECONSTRUCTION',)));self.assertEqual(r['terminal'],'abstained');np.testing.assert_array_equal(r['output'],x)
    def test_tool_request_has_bounded_branch(self):
        _,r=self.run_case(ScriptedClient(inspect=True));self.assertEqual(r['inspections'],1);self.assertIn('peer_inspection',[e['node'] for e in r['trace']])
    def test_no_ground_truth_fields(self):
        self.assertFalse({'clean','truth','true_building','attack'}&set(DefenseState.__annotations__))
    def test_strict_schema(self):
        with self.assertRaises(ValueError):strict_parse('{"operation":"PEER_CONSENSUS"}','planning')
        with self.assertRaises(ValueError):strict_parse('{"operation":"NO_OVERRIDE","justification":"x"}','planning')
    def test_conditional_bound_implies_recovery(self):
        rng=np.random.default_rng(51)
        accepted=0
        for _ in range(1000):
            q,x,c=rng.normal(size=3);d=rng.uniform(0,.3)
            if abs(x-q)>abs(c-q)+2*d:
                accepted+=1
                for truth in np.linspace(q-d,q+d,21):self.assertLess(abs(c-truth),abs(x-truth)+1e-12)
        self.assertGreater(accepted,100)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GraphTests))
    out=Path(__file__).parent/'validation';out.mkdir(exist_ok=True)
    save_json(out/'graph_integration.json',{'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'source':'scripted fixtures and mathematical checks, not Llama performance'})
    raise SystemExit(not result.wasSuccessful())
