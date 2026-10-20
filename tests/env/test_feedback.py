from sphyr.policies import ConstantFillPolicy
from sphyr.run_eval import open_session
from sphyr.run_feedback import run_episode


def test_feedback_trajectory_stores_original_task_for_future_rescoring():
    with open_session(sample_count=1, dataset_version="v2") as session:
        attempts = run_episode(session, ConstantFillPolicy("1"), "full_easy", 0, 1)
    assert len(attempts) == 2
    assert attempts[0]["prompt"] == attempts[1]["prompt"]
    assert "V" in attempts[0]["prompt"]
    assert attempts[0]["ground_truth"] == attempts[1]["ground_truth"]
    assert attempts[0]["optimal_compliance"] == attempts[1]["optimal_compliance"]
    assert all(a["scoring_policy"] == "masked-only-v1" for a in attempts)
