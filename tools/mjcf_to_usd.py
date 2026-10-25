"""Convert an MJCF to USD with the standalone Isaac Sim importer (no Kit).

With no arguments this preserves the original G1 conversion paths.
"""

import argparse
import os

from pxr import Plug

here = os.path.dirname(os.path.abspath(__file__))
import isaacsim.asset.importer.utils.impl as _iu

Plug.Registry().RegisterPlugins(os.path.join(os.path.dirname(_iu.__file__), "schemas"))
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--src",
    default=os.path.join(here, "..", "data", "g1_description", "g1_29dof.xml"),
)
parser.add_argument(
    "--dst",
    default=os.path.join(here, "..", "data", "g1_usd"),
)
args = parser.parse_args()
cfg = MJCFImporterConfig(
    mjcf_path=os.path.normpath(os.path.abspath(args.src)),
    usd_path=os.path.normpath(os.path.abspath(args.dst)),
)
print(MJCFImporter(cfg).import_mjcf())
