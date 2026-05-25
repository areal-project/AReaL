# Running AReaL with SkyPilot

This README includes examples and guidelines to running AReaL experiments with SkyPilot.
Make sure you have SkyPilot properly installed following
[our installation guide](../../docs/tutorial/installation.md#optional-install-skypilot)
before running this example. Note that all command lines shown in this file are assumed
to be execute under the root of AReaL repository.

## Running a Single Node Experiment

To run a single node experiment, you only need to setup the node with SkyPilot and
launch the experiment with AReaL local launcher.
[The following file](single_node.sky.yaml) shows a SkyPilot yaml that could launch a
simple GSM8K GRPO experiment in a single command line. This example is tested on both
GCP and a K8S cluster.

```yaml
name: areal-test-skypilot

resources:
  accelerators: A100:2
  autostop:
    idle_minutes: 10
    down: true
  cpus: 8+
  memory: 32GB+
  disk_size: 256GB
  image_id: docker:ghcr.io/areal-project/areal-runtime:v1.0.4-sglang

num_nodes: 1

file_mounts:
  /storage: # Should be consistent with the storage paths set in gsm8k_grpo_ray.yaml
    source: s3://my-bucket/  # or gs://, https://<azure_storage_account>.blob.core.windows.net/<container>, r2://, cos://<region>/<bucket>, oci://<bucket_name>
    mode: MOUNT  # MOUNT or COPY or MOUNT_CACHED. Defaults to MOUNT. Optional.

workdir: .

run: |
  python3 examples/math/gsm8k_rl.py \
    --config examples/math/gsm8k_grpo.yaml \
    scheduler.type=local \
    experiment_name=gsm8k-grpo \
    trial_name=trial0 \
    cluster.n_nodes=1 \
    cluster.n_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
    rollout.backend=sglang:d1 \
    actor.backend=fsdp:d1 \
    train_dataset.batch_size=4 \
    actor.mb_spec.max_tokens_per_mb=4096
```

To run the experiment, execute:

```bash
sky launch -c areal-test examples/skypilot/single_node.sky.yaml
```

To designate the cloud or infrastructure you wish to run your experiment on by adding
`--infra xxx`. For example:

```bash
sky launch -c areal-test examples/skypilot/single_node.sky.yaml --infra gcp
sky launch -c areal-test examples/skypilot/single_node.sky.yaml --infra aws
sky launch -c areal-test examples/skypilot/single_node.sky.yaml --infra k8s
```

## Running a Multi-Node Experiment

### Running AReaL with Ray Launcher

The following example shows how to setup a ray cluster with SkyPilot and then use AReaL
to run GRPO with GSM8K dataset on 2 nodes, each with 1 A100 GPU. This example is tested
on GCP and a K8S cluster.

Specify the resources and image used to run the experiment.

```yaml
resources:
  accelerators: A100:8
  image_id: docker:ghcr.io/areal-project/areal-runtime:v1.0.4-sglang
  memory: 256+
  cpus: 32+

num_nodes: 2

workdir: .
```

Designate shared storage. You could either use an existing cloud bucket or volume:

```yaml
file_mounts:
  /storage: # Should be consistent with the storage paths set in gsm8k_grpo_ray.yaml
    source: s3://my-bucket/  # or gs://, https://<azure_storage_account>.blob.core.windows.net/<container>, r2://, cos://<region>/<bucket>, oci://<bucket_name>
    mode: MOUNT  # MOUNT or COPY or MOUNT_CACHED. Defaults to MOUNT. Optional.
```

or create a new bucket or volume with SkyPilot:

```yaml
# Create an empty gcs bucket
file_mounts:
  /storage: # Should be consistent with the storage paths set in gsm8k_grpo_ray.yaml
    name: my-sky-bucket
    store: gcs  # Optional: either of s3, gcs, azure, r2, ibm, oci
```

For more information about shared storage with SkyPilot, check
[SkyPilot Cloud Buckets](https://docs.skypilot.co/en/latest/reference/storage.html) and
[SkyPilot Volume](https://docs.skypilot.co/en/latest/reference/volumes.html).

Next, prepare commands used to setup ray cluster and run the experiment.

```yaml
envs:
  EXPERIMENT_NAME: my-areal-experiment
  TRIAL_NAME: my-trial-name

run: |
  run: |
  # Get the Head node's IP and total number of nodes (environment variables injected by SkyPilot).
  head_ip=$(echo "$SKYPILOT_NODE_IPS" | head -n1)

  if [ "$SKYPILOT_NODE_RANK" = "0" ]; then
    echo "Starting Ray head node..."
    ray start --head --port=6379

    while [ $(ray status | grep node_ | wc -l) -lt $SKYPILOT_NUM_NODES ]; do
      echo "Waiting for all nodes to join... Current nodes: $(ray status | grep node_ | wc -l) / $SKYPILOT_NUM_NODES"
      sleep 5
    done

    echo "Executing training script on head node..."
    python3 examples/math/gsm8k_rl.py \
            --config examples/skypilot/gsm8k_grpo_ray.yaml \
            scheduler.type=ray \
            experiment_name=gsm8k-grpo \
            trial_name=trial0 \
            cluster.n_nodes=$SKYPILOT_NUM_NODES \
            cluster.n_gpus_per_node=$SKYPILOT_NUM_GPUS_PER_NODE \
            rollout.backend=sglang:d8 \
            actor.backend=fsdp:d8
  else
    sleep 10
    echo "Starting Ray worker node..."
    ray start --address $head_ip:6379
    sleep 5
  fi

  echo "Node setup complete for rank $SKYPILOT_NODE_RANK."
```

**Note**: If you are running on a cluster in which nodes are connected via infiniband,
you might need an additional config field to the example yaml file for the experiment to
run:

```yaml
config:
  kubernetes:
    pod_config:
      spec:
        containers:
        - securityContext:
            capabilities:
              add:
              - IPC_LOCK
```

### Launch the Ray Cluster and Run AReaL Experiment

Then you are ready to run AReaL with command line:

```bash
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml
```

To designate the cloud or infrastructure you wish to run your experiment on by adding
`--infra xxx`. For example:

```bash
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml --infra gcp
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml --infra aws
sky launch -c areal-test examples/skypilot/ray_cluster.sky.yaml --infra k8s
```

You should be able to see your AReaL running and producing training logs in your
terminal.

Successfully launched 2 nodes on GCP and deployed a ray cluster:
<img align="center" alt="Launching Ray Cluster" src="ray_launch.png" width="100%">

Successfully ran a training step:
<img align="center" alt="Running a train step" src="train_step_success.png" width="100%">

### Running AReaL with SkyPilot Launcher

AReaL plans to support a SkyPilot native launcher with
[SkyPilot Python SDK](https://docs.skypilot.co/en/latest/reference/api.html), which is
currently under development.
