"""GAL-014 test suite: leased host authority and start-cell semantics."""
import json
import unittest
from lab import START, HostRefusal
from start_semantics import SemanticsWorld,GoalLedger,Adapted,teacher_action,policy_action,episode,audit_row,train

class StartSemanticsTests(unittest.TestCase):
    def test_goal_and_exact_budget(self):
        for value in (False,True):
            w=SemanticsWorld(2130000,'tree',value,'test-goal')
            self.assertEqual(len(w.goal),33 if value else 32)
            self.assertEqual(START in w.goal,value)
            self.assertEqual(w.target,len(w.goal))
    def test_ledger_target_metadata(self):
        for flag in (False,True):
            w=SemanticsWorld(2130000,'tree',flag,'test-ledger')
            led=GoalLedger(flag);led.observe(w)
            self.assertEqual(START in led.targets(),flag)
    def test_task_aware_guard_preserves_host_offer(self):
        w=SemanticsWorld(2130000,'path',False,'test-guard')
        led=GoalLedger(False);led.observe(w)
        cp=[0.]*23;m=Adapted(cp)
        before=w.offer()
        choice,menu,token=policy_action(w,led,m,'greedy','conditional')
        self.assertEqual(before,(menu,token))
        self.assertNotIn(START,menu[choice])
        self.assertIn('STOP',menu)
    def test_host_rejects_stale_and_foreign(self):
        w=SemanticsWorld(2130000,'tree',False,'test-lease')
        menu,token=w.offer();selected=next(x for x in menu if x.startswith('INK_'))
        with self.assertRaises(HostRefusal):w.apply(selected,token+'x')
        w.apply(selected,token)
        with self.assertRaises(HostRefusal):w.apply(selected,token)
    def test_off_goal_start_consumes_budget(self):
        w=SemanticsWorld(2130000,'tree',False,'budget')
        self.assertEqual(w.target,32)
        w.own=set(w.goal)
        self.assertTrue(all(not n.startswith('INK_') for n in w.legal()))
    def test_teacher_observation_only(self):
        w=SemanticsWorld(2130000,'tree',False,'teacher')
        led=GoalLedger(False);led.observe(w)
        menu,_=w.offer();chosen=teacher_action(w,led,menu)
        self.assertNotIn(START,menu[chosen])
    def test_teacher_completes_both_objectives(self):
        m=Adapted([0.]*23)
        for seed in (2130000,2130001):
            for flag in (False,True):
                w,r=episode(seed,'tree',flag,m,'teacher')
                self.assertTrue(r['complete'],r)
                self.assertEqual(audit_row(r,w.trace),len(w.trace))
    def test_optimizer_is_real_and_deterministic(self):
        m1,a=train([0.]*23,8);m2,b=train([0.]*23,8)
        self.assertEqual(a,b)
        self.assertEqual(m1.adapter,m2.adapter)
        self.assertGreater(a['optimizer_updates'],0)
        self.assertTrue(any(abs(x)>0 for x in m1.adapter))

if __name__=='__main__':unittest.main()
