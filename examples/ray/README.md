# Role-specific Ray container images

Set `ray_runtime_env` on a role's `scheduling_spec` to run its Ray launcher and worker
processes in that environment. The default is empty and preserves the existing launcher
environment. This applies to the v1 controller as well.

```yaml
actor:
  scheduling_spec:
    - cpu: 4
      gpu: 1
      mem: 16
      ray_runtime_env:
        image_uri: registry.example/train:tag

rollout:
  scheduling_spec:
    - cpu: 4
      gpu: 4
      mem: 16
      ray_runtime_env:
        image_uri: registry.example/rollout:tag
```

Use an allocation mode and scheduling strategy appropriate for the experiment. For
colocation, AReaL places a separate launcher for the second role on each target node,
without reserving the GPUs again. Each worker receives its assigned GPU visibility mask.
Proxy workers inherit the rollout environment and use its Python interpreter. The
existing controller still manages model offload, reload and weight transfer; selecting
an image does not release GPU memory.

Ray's native image support uses Podman on the raylet node. The worker containers connect
to the existing raylet; they do not start additional raylets. If the raylet itself runs
in a container, that environment must support nested Podman. Ray documents privileged
permissions for this setup. A Kubernetes worker Pod must also provide suitable container
storage, GPU device access and shared mounts; a default restricted Pod is insufficient.
This does not create Kubernetes sidecar containers or require a Slurm service.

Prepare both images with the same Ray and Python versions as the raylet environment
(including the Python patch version) and an importable AReaL installation. Make model,
output and name-resolution paths available at the same locations wherever they are used.
Device access, CUDA driver libraries and shared IPC must also be configured for the
deployment.

For Ray versions that support `container.run_options`, custom Podman devices and mounts
can be specified instead of `image_uri`:

```yaml
ray_runtime_env:
  container:
    image: registry.example/rollout:tag
    run_options:
      - --volume=/shared:/shared
      - --device=/dev/nvidia0
```

This is an illustrative mount/device entry, not a complete GPU setup. Check the
installed Ray version's supported fields and Podman/NVIDIA configuration.

All scheduling specifications within a role must have the same runtime environment,
environment variables and CPU/GPU/memory requests. Explicit `env_vars` on the scheduling
specification override `ray_runtime_env.env_vars`. With image runtime environments,
AReaL does not copy the driver's generated `PYTHONPATH`; set an explicit path only if
the corresponding source is mounted inside that image.

See Ray's
[runtime environment API](https://docs.ray.io/en/latest/ray-core/api/doc/ray.runtime_env.RuntimeEnv.html)
and
[container deployment guide](https://docs.ray.io/en/latest/serve/advanced-guides/multi-app-container.html).
