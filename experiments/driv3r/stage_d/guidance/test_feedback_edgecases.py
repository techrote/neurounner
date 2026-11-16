import unittest
from hashlib import sha256
from experiments.driv3r.stage_d.guidance.feedback_audit import seal_case,audit_shadow_cases
S=sha256(b'test').hexdigest()
def case(name,index,b=(10,0,3),a=(8,1,3),choice='UNDO',eligible=True,receipt=True):
    return seal_case(dict(case_id=name,decision_index=index,scenario_sha256=S,
        before=dict(snapshot_id=name+'b',goal_score=b[0],violations=b[1],progress_cells=b[2]),
        after=dict(snapshot_id=name+'a',goal_score=a[0],violations=a[1],progress_cells=a[2]),
        hypothetical_undo_eligible=eligible,hypothetical_receipt_available=receipt,shadow_choice=choice))
class EdgeCases(unittest.TestCase):
    def test_oracle_stop_is_false_undo(self):
        d=audit_shadow_cases([case('u',1,eligible=False)])
        self.assertEqual(d['errors']['false_undo'],1)
        self.assertEqual(d['errors']['inadmissible_undo'],1)
        self.assertEqual(d['confusion']['STOP']['UNDO'],1)
    def test_missing_receipt_is_false_undo(self):
        d=audit_shadow_cases([case('u',1,receipt=False)])
        self.assertEqual(d['errors']['false_undo'],1)
        self.assertEqual(d['errors']['inadmissible_undo'],1)
    def test_true_reference_undo_not_false(self):
        d=audit_shadow_cases([case('u',1)])
        self.assertEqual(d['errors']['false_undo'],0)
    def test_retried_identical_metric_pair_new_ids_is_risk(self):
        d=audit_shadow_cases([case('u1',1),case('u2',2)])
        self.assertEqual(d['errors']['oscillation_risk'],1)
    def test_distinct_metric_pair_not_risk(self):
        d=audit_shadow_cases([case('u1',1),case('u2',2,a=(7,1,3))])
        self.assertEqual(d['errors']['oscillation_risk'],0)
    def test_admissible_but_unwarranted_undo_false_only(self):
        d=audit_shadow_cases([case('u',1,a=(11,0,4))])
        self.assertEqual(d['errors']['false_undo'],1)
        self.assertEqual(d['errors']['inadmissible_undo'],0)
if __name__=='__main__':unittest.main()
