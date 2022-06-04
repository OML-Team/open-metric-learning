from oml.models.utils import remove_prefix_from_state_dict


def test_remove_prefix_from_state_dict() -> None:
    inp = {"model.layer4": 1}, "layer4"
    gt = {"layer4": 1}
    ans = remove_prefix_from_state_dict(*inp)  # type: ignore

    assert gt == ans, ans

    inp = {"layer4": 1}, "layer4"
    gt = {"layer4": 1}
    ans = remove_prefix_from_state_dict(*inp)  # type: ignore

    assert gt == ans, ans
