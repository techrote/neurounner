#!/usr/bin/env python3
"""W5 six-way loss target placement tests. No model download or optimizer."""
from worm_train_v5 import first_target_position
def check():
    assert first_target_position([-100,-100,32,5])==2
    assert first_target_position([-100,36,5])==1
    for invalid in ([],[-100,-100],[-100,5],[-100,2,5],[32,5],[-100,32,4]):
        try: first_target_position(invalid)
        except ValueError: pass
        else: raise AssertionError("invalid target framing accepted "+str(invalid))
    print("W5 full masked prompt, legal lexical class target and EOS5 PASS")
if __name__=="__main__": check()
