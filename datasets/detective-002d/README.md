# DETECTIVE-002D LFM v5 chat-template API requalification

Focused follow-up to DETECTIVE-002C. Same exact LFM2.5-350M source, same exact Transformers 5.20.1 wheel and same B01-B04 prompts. The only intended change is explicit v5 chat-template return handling: text render with return_dict=False, then tokenized mapping with return_dict=True and input_ids access. Model weight is downloaded only after both paths pass. No training/selection/license/admission.
