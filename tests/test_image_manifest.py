import json
from pathlib import Path


def test_energy_image_is_pinned() -> None:
    manifest = json.loads((Path(__file__).parents[1] / "image-manifest.json").read_text())
    assert manifest["tasks"]["energy-ac-optimal-power-flow"]["image"] == (
        "public.ecr.aws/z4x6c5k7/skillsbench@sha256:2d96fb4b515ae40638c0046740cfb2f5e6d58248b9dbfbe62ad16893caf1f7f3"
    )
