import copy
import json
import unittest
from .history_motif_lab import Adapter,episode
from .verify_history_motif import verify_episode
from .world import specs

class HistoryVerificationTests(unittest.TestCase):
    def test_policy_probabilities_are_verified(self):
        m=Adapter(575,False);s=specs('development',1)[0]
        o=episode(s,'arc',m,36)
        self.assertEqual(verify_episode(o,m).state_sha(),o['final_sha256'])
        fake=copy.deepcopy(o)
        first=next(iter(fake['probabilities'][0]['distribution']))
        fake['probabilities'][0]['distribution'][first]=.99
        with self.assertRaises(ValueError):verify_episode(fake,m)
    def test_prevent_false_native_receipt(self):
        m=Adapter(575,False);s=specs('development',1)[0]
        fake=episode(s,'arc',m,36)
        fake['native_receipts']=1
        with self.assertRaises(ValueError):verify_episode(fake,m)
if __name__=='__main__':unittest.main()
