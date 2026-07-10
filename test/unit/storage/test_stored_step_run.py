from gbserver.storage.stored_step_run import StoredStepRun


def test_skypilot_handle_defaults_to_none():
    run = StoredStepRun(build_id="b1", target_id="t1", definition_uri="file:///x")
    assert run.skypilot_handle is None


def test_skypilot_handle_round_trips_dict():
    handle = {"cluster_name": "gb-cma-host-b-t-s0-abcd1234", "job_id": 1, "done_marker": "/w/.gb_done_s0"}
    run = StoredStepRun(build_id="b1", target_id="t1", definition_uri="file:///x", skypilot_handle=handle)
    assert run.skypilot_handle == handle
