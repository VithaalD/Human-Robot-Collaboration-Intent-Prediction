import json
from types import SimpleNamespace
import pytest
from frame_sampling import validate_training_timing
def dataset(tmp_path,pilot=True,same_source=False):
 protocol='within_session_trial_pilot' if pilot else 'session_held_out'
 clips={n:{'source_recording_id':'one' if same_source else n,'session_id':'real_session','split_group_id':'real_session'} for n in ['a','b']}
 annotation={'sample_fps':8,'annotations_confirmed':True,'split_protocol':protocol,'database':{
 n:{'subset':s,'source_recording_id':clips[n]['source_recording_id'],'split_group_id':'real_session'}
 for n,s in [('a','training'),('b','validation')]}}
 (tmp_path/'sampling.json').write_text(json.dumps({'sample_fps':8,'split_protocol':protocol,'clips':clips}))
 (tmp_path/'annotations.json').write_text(json.dumps(annotation))
 return SimpleNamespace(sample_fps=8,video_path=tmp_path,annotation_path=tmp_path/'annotations.json',allow_within_session_pilot=False)
def test_strict_default_still_rejects_shared_session(tmp_path):
 with pytest.raises(ValueError,match='session cannot span'):validate_training_timing(dataset(tmp_path))
def test_explicit_pilot_accepts_distinct_trials(tmp_path):
 opt=dataset(tmp_path);opt.allow_within_session_pilot=True;validate_training_timing(opt)
def test_pilot_never_allows_source_trial_leakage(tmp_path):
 opt=dataset(tmp_path,same_source=True);opt.allow_within_session_pilot=True
 with pytest.raises(ValueError,match='source recording cannot span'):validate_training_timing(opt)
def test_pilot_requires_explicit_dataset_marking(tmp_path):
 opt=dataset(tmp_path,pilot=False);opt.allow_within_session_pilot=True
 with pytest.raises(ValueError,match='explicitly marked'):validate_training_timing(opt)
