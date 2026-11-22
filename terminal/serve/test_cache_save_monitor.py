import io,json,queue,threading,time,unittest,subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from serve.server import StrataEngine,Service,MockEngine,ByteTokenizer
from serve.frontend import ChatTemplate
ROOT=Path(__file__).resolve().parents[1]
class CacheSaveMonitor(unittest.TestCase):
 def service(self):
  tok=ByteTokenizer();eng=MockEngine(tok,'ok',max_context=4096)
  eng.disk_cache_active=True
  svc=Service(eng,tok,ChatTemplate(ROOT/'serve/chat_template.jinja'))
  svc.status.update(busy=True,first_token=1000.,started=1000.,generated=1077)
  return svc
 def test_save_is_separate_and_does_not_reuse_decaying_speed(self):
  svc=self.service()
  with patch('serve.server.time.time',return_value=1030.):
   self.assertTrue(svc._saving_cache());self.assertEqual(svc._tok_s(),0.)
   live=svc.metrics()['live']
  self.assertEqual(live['state'],'saving');self.assertIsNone(live['tok_s'])
  self.assertIsNone(live['tok_s_mean']);self.assertEqual(live['generated'],1077)
 def test_restore_before_first_token_is_still_prefill(self):
  svc=self.service();svc.status['first_token']=None
  self.assertFalse(svc._saving_cache());self.assertEqual(svc.metrics()['live']['state'],'reading')
 def test_idle_and_parallel_are_not_misclassified(self):
  svc=self.service();svc.status['busy']=False
  self.assertFalse(svc._saving_cache())
  svc.status['busy']=True;svc.engine.batch=2
  self.assertFalse(svc._saving_cache())
 def test_real_protocol_transitions_and_completion(self):
  e=StrataEngine('/unused',[],lazy=True)
  e.ended=False;e.unloaded=False;e.proc=SimpleNamespace(stdin=io.StringIO(),poll=lambda:None)
  e.lines=queue.Queue()
  for line in ['DISK_CACHE 0 100','T 65','DISK_CACHE 1 100','DISK_CACHE 100 100','DONE 2 10 100 20 stop']:
   e.lines.put(line)
  gen=e.generate([1],2,{},threading.Event())
  self.assertIsNone(next(gen));self.assertTrue(e.disk_cache_active)
  self.assertEqual(next(gen),65);self.assertFalse(e.disk_cache_active)
  self.assertIsNone(next(gen));self.assertTrue(e.disk_cache_active)
  self.assertIsNone(next(gen));self.assertTrue(e.disk_cache_active)
  self.assertEqual(list(gen),[]);self.assertFalse(e.disk_cache_active)
  self.assertEqual(e.last['decode_ms'],20.)
 def test_failed_save_does_not_stick_on_next_request(self):
  e=StrataEngine('/unused',[],lazy=True)
  e.ended=False;e.unloaded=False;e.proc=SimpleNamespace(stdin=io.StringIO(),poll=lambda:None)
  e.lines=queue.Queue()
  for line in ['T 65','DISK_CACHE 1 100',None]:e.lines.put(line)
  gen=e.generate([1],2,{},threading.Event());next(gen);next(gen)
  with self.assertRaises(Exception):next(gen)
  self.assertFalse(e.disk_cache_active)
 def test_spark_leaves_gaps_and_preserves_real_zero(self):
  text=(ROOT/'serve/web/app.js').read_text();start=text.index('function spark(');end=text.index('\nfunction setMetric(',start)
  script='const out={}; const $=()=>({querySelector:s=>({setAttribute:(k,v)=>out[s]=v})});\n'+text[start:end]+'''\nspark('x',[60,60,null,null,50,50],undefined,true);const gap={...out};
  spark('x',[60,60,0,0],undefined,true);console.log(JSON.stringify({gap,zero:out}));'''
  out=json.loads(subprocess.check_output(['node','-e',script],text=True))
  self.assertEqual(out['gap']['.line'].count('M'),2)
  self.assertNotIn('NaN',out['gap']['.line']);self.assertTrue(out['zero']['.line'].endswith('100.00,30.00'))
if __name__=='__main__':unittest.main(verbosity=2)
