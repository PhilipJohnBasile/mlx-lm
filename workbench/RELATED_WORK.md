# Related work already stored in GitHub

The core-MLX changes belong in `PhilipJohnBasile/mlx`, not the MLX-LM source tree.
Their pinned branches are preserved and should not be conflated with model-speed
experiments. The launcher source and evidence are also copied in this handoff.

| Work | Branch | Commit |
|---|---|---|
| Launcher worker failure exit status | `fix/mlx-launch-exit-status` | `32136cc8ed4d888d3eed50822ea2332e52cff64b` |
| JACCL mesh peer-loss fix | `fix/jaccl-mesh-peer-loss-3910` | `e146fca163942b507ef41726655f9908007dd9ec` |
| JACCL investigation and regression workflows | `test/jaccl-peer-loss-3910` | `732a5cea4beef466fc172cd3e20df65f772370eb` |
| MLX source baseline | upstream commit | `b6368984b8e02a3fb3ee7986846c0fb85e1fccf7` |

Clone a separate core-MLX working directory without changing this checkout:

```sh
git clone https://github.com/PhilipJohnBasile/mlx.git mlx-core-work
cd mlx-core-work
git switch fix/mlx-launch-exit-status
# Other independent lines of work:
# git switch fix/jaccl-mesh-peer-loss-3910
# git switch test/jaccl-peer-loss-3910
```

The earlier MTPLX PR conflict repairs remain in their original branches and PRs
in `youssofal/MTPLX` / `PhilipJohnBasile/MTPLX`; they are not silently merged into
this MLX-LM optimization workbench. This handoff does not change those PRs.

The source-only workbench retains compilation logs, source, and binary hashes.
AIR/metallib and temporary sanitizer executables are reproducible build outputs,
not required editable source. Original hosted artifacts were:
- MLX source archive: PhilipJohnBasile/mlx run 33933716033, artifact 9959434248.
- GDN compiler run 33943739873, artifact 9962651445.
- GDN compiler/host run 33944434070, artifact 9962873315.
- GDN hosted native run 33944537249, artifact 9962907100.

Hosted artifacts have retention limits; the important source and logs have been
copied into the workbench rather than relying on those expiring links.
