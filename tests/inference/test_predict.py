from brats2026.inference.predict import predict_command, predict_ensemble_commands
from brats2026.nnunet.plan import GOAT_TRAINER, RESENC_L_PLANS


def test_predict_command_flags_and_paths():
    cmd = predict_command("modelA", "/in", "/out")
    assert cmd[0] == "nnUNetv2_predict"
    assert cmd[cmd.index("-i") + 1] == "/in"
    assert cmd[cmd.index("-o") + 1] == "/out"
    assert cmd[cmd.index("-d") + 1] == "modelA"
    assert cmd[cmd.index("-tr") + 1] == GOAT_TRAINER
    assert cmd[cmd.index("-p") + 1] == RESENC_L_PLANS
    assert cmd[cmd.index("-c") + 1] == "3d_fullres"


def test_predict_command_folds_expand():
    cmd = predict_command("m", "/in", "/out", folds=(0, 1, 2, 3, 4))
    fi = cmd.index("-f")
    assert cmd[fi + 1 : fi + 6] == ["0", "1", "2", "3", "4"]


def test_predict_command_tta_default_off_flag():
    assert "--disable_tta" not in predict_command("m", "/in", "/out")


def test_predict_command_disable_tta_adds_flag():
    assert "--disable_tta" in predict_command("m", "/in", "/out", disable_tta=True)


def test_ensemble_one_command_per_member():
    members = ["m0", "m1", "m2"]
    cmds = predict_ensemble_commands(members, "/in", "/out_root")
    assert len(cmds) == 3
    for member, cmd in zip(members, cmds):
        out = cmd[cmd.index("-o") + 1]
        assert out.endswith(member)
        assert cmd[cmd.index("-d") + 1] == member


def test_ensemble_passes_through_kwargs():
    cmds = predict_ensemble_commands(["m0"], "/in", "/out", disable_tta=True, folds=(0,))
    assert "--disable_tta" in cmds[0]
    fi = cmds[0].index("-f")
    assert cmds[0][fi + 1] == "0"
