"""Explicit host placement must count full mirrors, preserve disk roots and keep Auto unchanged."""
import math
import unittest
from strata_control import split_engine as S

class HostMemory(unittest.TestCase):
    def model(self):
        return S.ModelFacts({'IQ3_XXS':144},48*S.GIB,'Q6_K',{'Q6_K':144},28*S.GIB,48,512,
                            output_type='Q6_K',ple_ram_bytes=27*S.GIB)
    def decide(self,**kw):
        options=dict(enabled=True,system='linux',cards=[S.Card(i,'V100',16384,70) for i in range(2)],
                     ram_total_gib=55,ram_available_gib=50,model=self.model(),allow_untested=True,cache_mib=256)
        options.update(kw);return S.decide(**options)
    def test_fixed_placement_fits_but_full_mirror_does_not(self):
        fixed=self.decide(expert_ram_gib=36,fixed_experts=True,ple_direct=True)
        self.assertTrue(fixed.use_split,fixed.reasons)
        dynamic=self.decide(expert_ram_gib=36,ple_direct=True)
        self.assertFalse(dynamic.use_split)
        self.assertGreater(dynamic.need_ram_gib,fixed.need_ram_gib)
        self.assertFalse(self.decide(expert_ram_gib=36,fixed_experts=True).use_split)
        self.assertFalse(self.decide().use_split)
    def test_invalid_memory_values(self):
        for value in (math.nan,math.inf,-1):
            self.assertFalse(self.decide(expert_ram_gib=value).use_split)
        self.assertFalse(self.decide(fixed_experts=True).use_split)
        with self.assertRaises(ValueError): S._memory_options({'split_expert_ram_gib':36},'auto')
    def test_run_config_preserves_each_root_and_fixed_placement(self):
        d=self.decide(expert_ram_gib=36,fixed_experts=True,ple_direct=True)
        ple=['--ple-io','direct','--ple-row-cache','65536','--ple-disk-cache','/disk/a','100','--ple-disk-cache','/disk/b','200']
        base={'args':ple,'split_expert_ram_gib':36,'split_fixed_experts':True,'env':{'OTHER':'kept'}}
        cfg=S.split_config(base,d,exe='/engine/strata-split-2',native='/model',mtp_dir='/mtp',draft_vocab='/vocab',context=8192)
        self.assertEqual(cfg['args'][-len(ple):],ple)
        self.assertEqual(cfg['env']['FNX_FIXED_EXPERTS'],'1')
        self.assertEqual(cfg['env']['FNX_EXPERT_RAM_GIB'],'36.0')
        self.assertEqual(cfg['env']['FNX_ARENA'],'0')
        self.assertEqual(cfg['env']['OTHER'],'kept')
        self.assertEqual(cfg['split_fallback']['args'],ple)
    def test_explicit_on_does_not_require_auto_speed_claim(self):
        d=self.decide(expert_ram_gib=36,fixed_experts=True,ple_direct=True)
        cards=[S.Card(i,'V100',16384,70) for i in range(2)]
        self.assertIsNone(S._auto_gates(d,cards,self.model(),'auto','on'))
        self.assertIsNotNone(S._auto_gates(d,cards,self.model(),'auto','auto'))
    def test_early_plan_and_launch_agree_on_explicit_v100(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            native=Path(tmp)/'model.gguf';native.touch()
            (Path(tmp)/'draft_vocab_en.bin').touch()
            cfg={'split_engine':'on','split_expert_ram_gib':36,'split_fixed_experts':True,
                 'split_conversation_cache_mib':256,'parallel':1,'cwd':tmp,
                 'args':['--native',str(native),'--mtp',tmp,'--kv','int8','--max-context','8192','--ple-io','direct']}
            hooks=dict(system='linux',probe=lambda _: [S.Card(i,'V100',16384,70) for i in range(2)],
                       ram=lambda:(55,50),scan=lambda _:self.model(),exe_for_cards=lambda *_:'/engine/strata-split-2')
            self.assertTrue(S.expected(cfg,native=str(native),**hooks))
            decision=S.apply_to_config(cfg,data_dir=tmp,**hooks)
            self.assertTrue(decision.use_split,decision.reasons)
            self.assertEqual(cfg['env']['FNX_FIXED_EXPERTS'],'1')
            self.assertEqual(cfg['args'][cfg['args'].index('--conversation-cache-mib')+1],'256')
