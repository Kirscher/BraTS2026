from brats2026.nnunet.plan import (
    GOAT_TRAINER,
    RESENC_L_PLANNER,
    RESENC_L_PLANS,
    kfold_train_commands,
    nnunet_env,
    plan_and_preprocess_command,
    train_command,
)


def test_nnunet_env_roots_under_work(tmp_path):
    env = nnunet_env(tmp_path / "work")
    assert env["nnUNet_raw"].endswith("work/nnUNet_raw")
    assert env["nnUNet_preprocessed"].endswith("work/nnUNet_preprocessed")
    assert env["nnUNet_results"].endswith("work/nnUNet_results")


def test_plan_and_preprocess_uses_resenc_l_and_verify():
    cmd = plan_and_preprocess_command(501)
    assert cmd[0] == "nnUNetv2_plan_and_preprocess"
    assert "-d" in cmd and "501" in cmd
    assert RESENC_L_PLANNER in cmd
    assert "--verify_dataset_integrity" in cmd
    assert "3d_fullres" in cmd


def test_plan_command_can_drop_verify():
    assert "--verify_dataset_integrity" not in plan_and_preprocess_command(verify=False)


def test_train_command_wires_trainer_and_plans():
    cmd = train_command(fold=0)
    assert cmd[:4] == ["nnUNetv2_train", "501", "3d_fullres", "0"]
    assert cmd[cmd.index("-tr") + 1] == GOAT_TRAINER
    assert cmd[cmd.index("-p") + 1] == RESENC_L_PLANS


def test_kfold_train_commands_count_and_folds():
    cmds = kfold_train_commands(folds=5)
    assert len(cmds) == 5
    assert [c[3] for c in cmds] == ["0", "1", "2", "3", "4"]
