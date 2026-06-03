from brats2026.tasks import TASKS, get_task


def test_all_five_tasks_are_configured():
    assert set(TASKS) == {"task1", "task2", "task3", "task4", "task5"}


def test_goat_rule_note_is_visible_in_task_config():
    task = get_task("task3")

    assert task.kind == "mri_segmentation"
    assert task.label_map == {"NCR": 1, "ED": 2, "ET": 3}
    assert "forbid" in task.notes
    assert "BraTS-GoAT" in task.notes
