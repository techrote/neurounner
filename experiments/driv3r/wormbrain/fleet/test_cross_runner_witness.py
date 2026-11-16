"""Cross-host clock-epoch and fabricated four-worm overlap rejection tests.

All samples below are MOCK.  They establish no inference/Godot/CPU capacity.
"""
import copy
from datetime import datetime,timezone
from email.utils import format_datetime
import unittest
from cross_runner_witness import SCHEMA, URL, NS, audit, bound, offsets

UTC = 1_800_000_000 * NS

def date(t):
    return format_datetime(datetime.fromtimestamp(t,timezone.utc),usegmt=True)

def runner(job, epoch, shift=0):
    # A distinct monotonic epoch on every simulated GitHub VM.
    def sample(sec):
        return {"url":URL,"transport":"HTTPS","http_date":date(1_800_000_000+sec),
                "before_monotonic_ns":epoch+(sec*NS)+100_000_000,
                "after_monotonic_ns":epoch+(sec*NS)+200_000_000}
    # utc = 1800000000 s + (monotonic - epoch)
    return {"schema":SCHEMA,"job_id":str(job),
            "date_samples":[sample(10),sample(11)],
            "episodes":[{"job_id":str(job),"lease_id":"lease-long-"+str(job)+"-"+str(i),
                "pid":1000+int(job)*2+i,"status":"completed",
                "native_execution":False,"training":False,"optimizer_steps":0,
                "source_sha256":"a"*64,"model_sha256":"b"*64,
                "original_artifact_sha256":"c"*64,"model_revision":"pinned-original-sha",
                "model_forwards":8,"forward_start_ns":epoch+(20+shift)*NS,
                "forward_end_ns":epoch+(35+shift)*NS} for i in range(2)]}

class TestCrossRunner(unittest.TestCase):
    def test_common_time_distinct_monotonic_epochs(self):
        result=audit([runner(1,100*NS),runner(2,10_000_000*NS)])
        self.assertTrue(result["producer_claimed_four_episode_overlap"])
        self.assertGreater(result["producer_claimed_bounded_utc_episode_overlap_seconds"],10)
        self.assertEqual(result["independently_verified_active_models"],0)
        self.assertFalse(result["four_simultaneous_model_forward_calls_proven"])

    def test_nonoverlap_in_common_time_is_not_four_active(self):
        result=audit([runner(1,100*NS),runner(2,10_000_000*NS,shift=30)])
        self.assertFalse(result["producer_claimed_four_episode_overlap"])
        self.assertEqual(result["producer_claimed_bounded_utc_episode_overlap_seconds"],0)

    def test_local_residency_must_overlap(self):
        rr=[runner(1,100*NS),runner(2,10_000_000*NS)]
        rr[0]["episodes"][1]["forward_start_ns"]+=20*NS
        rr[0]["episodes"][1]["forward_end_ns"]+=20*NS
        with self.assertRaises(ValueError):
            audit(rr)

    def test_no_shared_clock_if_date_sample_missing_or_inconsistent(self):
        r=runner(1,100*NS)
        with self.assertRaises(ValueError):
            offsets(r["date_samples"][:1])
        r["date_samples"][1]["http_date"]=date(1_800_000_500)
        with self.assertRaises(ValueError):
            offsets(r["date_samples"])

    def test_reject_forged_or_duplicate_process_evidence(self):
        cases=[
            lambda r:r[1].__setitem__("job_id","1"),
            lambda r:r[1]["episodes"][0].__setitem__("lease_id",r[0]["episodes"][0]["lease_id"]),
            lambda r:r[0]["episodes"][1].__setitem__("pid",r[0]["episodes"][0]["pid"]),
            lambda r:r[0]["episodes"][0].__setitem__("optimizer_steps",1),
            lambda r:r[0]["episodes"][0].__setitem__("native_execution",True),
            lambda r:r[0]["episodes"][0].__setitem__("source_sha256","f"*64),
            lambda r:r[0]["episodes"][0].__setitem__("model_forwards",0),
            lambda r:r[0]["episodes"][0].__setitem__("status","queued")
        ]
        for change in cases:
            rr=[runner(1,100*NS),runner(2,10_000_000*NS)]
            change(rr)
            with self.subTest(change=str(change)), self.assertRaises(ValueError):
                audit(rr)

    def test_no_monotonic_without_https_date(self):
        s=runner(1,100*NS)["date_samples"][0]
        for key,value in [("transport","HTTP"),("url","https://example.org/"),
                          ("after_monotonic_ns",s["before_monotonic_ns"]+5*NS)]:
            changed=copy.deepcopy(s)
            changed[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                bound(changed)

if __name__=="__main__":
    unittest.main()
