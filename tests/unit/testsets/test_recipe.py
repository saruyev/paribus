"""Tests for testsets.recipe: what a valid recipe looks like and which mistakes fail."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from testsets.recipe import Codec, Generate, Mix, Operation, Pad, Resample, load_recipe, register_op

_SPEECH = {"source": {"type": "folder", "root": "speech", "license": "CC-BY-4.0"}, "take": 5}
_BACKGROUNDS = {"music": {"source": {"type": "folder", "root": "music", "license": "CC-BY-3.0"}}}


def _recipe(tmp_path: Path, **overrides: Any) -> Path:
    """Write a small valid recipe, with top-level keys replaced by `overrides`."""
    recipe = {
        "name": "t0",
        "seed": 13,
        "speech": [_SPEECH],
        "backgrounds": _BACKGROUNDS,
        "conditions": [{"name": "clean"}],
        **overrides,
    }
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(recipe), encoding="utf-8")
    return path


def _conditions(tmp_path: Path, *conditions: dict[str, Any]) -> Path:
    """A recipe with only the given conditions."""
    return _recipe(tmp_path, conditions=list(conditions))


def test_operations_parse_from_their_yaml_form(tmp_path: Path) -> None:
    """One-key mappings and bare shorthand values both become typed operations."""
    path = _conditions(
        tmp_path,
        {
            "name": "phone",
            "ops": [
                {"mix": {"background": "music", "snr_db": [0, 10]}},
                {"pad": {"before": 2, "fill": "floor"}},
                {"resample": 8000},
                {"codec": "pcm_mulaw"},
            ],
        },
    )

    (condition,) = load_recipe(path).conditions

    assert condition.ops == [
        Mix(background="music", snr_db=(0, 10)),
        Pad(before=2, fill="floor"),
        Resample(rate=8000),
        Codec(name="pcm_mulaw"),
    ]


def test_a_no_speech_condition_starts_with_generate(tmp_path: Path) -> None:
    """Clips without speech are generated from a fill, `count` of them."""
    path = _conditions(
        tmp_path,
        {"name": "only_music", "speech": "none", "count": 8, "ops": [{"generate": "music"}]},
    )

    with pytest.raises(ValidationError, match="seconds"):
        load_recipe(path)

    path = _conditions(
        tmp_path,
        {
            "name": "only_music",
            "speech": "none",
            "count": 8,
            "ops": [{"generate": {"fill": "music", "seconds": [5, 15]}}],
        },
    )
    (condition,) = load_recipe(path).conditions
    assert condition.ops == [Generate(fill="music", seconds=(5, 15))]


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        ({"name": "x", "ops": [{"generate": {"fill": "zero", "seconds": 5}}]}, "only for"),
        ({"name": "x", "speech": "none", "count": 2, "ops": []}, "start with generate"),
        (
            {"name": "x", "speech": "none", "ops": [{"generate": {"fill": "zero", "seconds": 1}}]},
            "needs count",
        ),
        (
            {"name": "x", "ops": [{"mix": {"background": "rain", "snr_db": 0}}]},
            "unknown background 'rain'",
        ),
        ({"name": "x", "ops": [{"pad": {"fill": "rain"}}]}, "unknown background 'rain'"),
        ({"name": "x", "ops": [{"mix": {"background": "music", "snr": 0}}]}, "snr_db"),
        ({"name": "x", "ops": [{"reverb": {}}]}, "unknown operation 'reverb'"),
        ({"name": "x", "ops": [{"mix": {}, "pad": {}}]}, "one key"),
        ({"name": "x", "ops": [{"pad": 3}]}, "needs a mapping"),
        ({"name": "x", "ops": [{"gain": {"db": [3, -3]}}]}, "low end 3"),
        ({"name": "x", "ops": [{"codec": {"name": "pcm_mulaw", "bitrate": 8000}}]}, "opus only"),
        ({"name": "x y"}, "pattern"),
    ],
)
def test_condition_mistakes_fail(tmp_path: Path, condition: dict[str, Any], message: str) -> None:
    """Each mistake fails at load time with a message that names it."""
    with pytest.raises(ValidationError, match=message):
        load_recipe(_conditions(tmp_path, condition))


def test_condition_names_must_be_unique(tmp_path: Path) -> None:
    """Two conditions with one name would write to the same folder."""
    path = _conditions(tmp_path, {"name": "clean"}, {"name": "clean"})

    with pytest.raises(ValidationError, match="repeat: clean"):
        load_recipe(path)


def test_background_cannot_shadow_a_built_in_fill(tmp_path: Path) -> None:
    """A background called `zero` would make `fill: zero` ambiguous."""
    path = _recipe(tmp_path, backgrounds={"zero": _BACKGROUNDS["music"]})

    with pytest.raises(ValidationError, match="reserved"):
        load_recipe(path)


def test_background_license_filter_needs_both_settings(tmp_path: Path) -> None:
    """An allow-list without the column to check it against would filter nothing."""
    backgrounds = {"music": {**_BACKGROUNDS["music"], "licenses": ["CC-BY-3.0"]}}

    with pytest.raises(ValidationError, match="go together"):
        load_recipe(_recipe(tmp_path, backgrounds=backgrounds))


def test_source_configs_are_validated_like_build_source(tmp_path: Path) -> None:
    """A typo inside a source config fails when the recipe is loaded, not mid-build."""
    speech = {**_SPEECH, "source": {**_SPEECH["source"], "rooot": "x"}}

    with pytest.raises(ValidationError, match="rooot"):
        load_recipe(_recipe(tmp_path, speech=[speech]))


def test_duration_filter_must_be_ordered(tmp_path: Path) -> None:
    """A duration filter of [20, 3] would keep nothing."""
    with pytest.raises(ValidationError, match="duration"):
        load_recipe(_recipe(tmp_path, speech=[{**_SPEECH, "duration": [20, 3]}]))


def test_concat_settings_are_validated(tmp_path: Path) -> None:
    """A concat block with a reversed length range fails at load time."""
    speech = {**_SPEECH, "concat": {"seconds": [300, 30]}}

    with pytest.raises(ValidationError, match="low end 300"):
        load_recipe(_recipe(tmp_path, speech=[speech]))


def test_an_operation_name_cannot_be_registered_twice() -> None:
    """A plugin cannot silently replace a built-in operation or another plugin's."""
    with pytest.raises(ValueError, match="'mix' is already registered"):

        @register_op("mix")
        class _Mine(Operation):
            """Another `mix`."""


def test_plugins_next_to_the_recipe_register_their_operations(tmp_path: Path) -> None:
    """A module beside the recipe is imported first, so its operation can be used."""
    (tmp_path / "recipe_test_plugin.py").write_text(
        "from testsets import Operation, register_op\n"
        "\n"
        "@register_op('test_plugin_op')\n"
        "class Nothing(Operation):\n"
        "    level: float = 0.0\n",
        encoding="utf-8",
    )
    path = _recipe(
        tmp_path,
        plugins=["recipe_test_plugin"],
        conditions=[{"name": "x", "ops": [{"test_plugin_op": {"level": 2}}]}],
    )

    recipe = load_recipe(path)

    assert recipe.plugins == ["recipe_test_plugin"]
    assert type(recipe.conditions[0].ops[0]).__name__ == "Nothing"


def test_a_missing_plugin_names_itself(tmp_path: Path) -> None:
    """A typo in `plugins` says which module and where it was looked for."""
    path = _recipe(tmp_path, plugins=["no_such_plugin_here"])

    with pytest.raises(ImportError, match="'no_such_plugin_here' not found"):
        load_recipe(path)


def test_an_error_inside_a_plugin_is_not_hidden(tmp_path: Path) -> None:
    """A plugin that exists but imports something missing reports the real cause."""
    (tmp_path / "recipe_broken_plugin.py").write_text("import no_such_dependency\n")
    path = _recipe(tmp_path, plugins=["recipe_broken_plugin"])

    with pytest.raises(ModuleNotFoundError, match="no_such_dependency"):
        load_recipe(path)
