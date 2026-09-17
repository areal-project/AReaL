from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api import StepInfo
from areal.trainer.sft_trainer import _restore_sampler_epoch_for_recovery
from areal.utils.data import cycle_dataloader
from areal.utils.recover import RecoverInfo


def test_cycle_dataloader_preserves_restored_sampler_epoch():
    dataset = list(range(16))
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=7,
    )
    sampler.set_epoch(3)
    dataloader = StatefulDataLoader(dataset, batch_size=4, sampler=sampler)

    data_generator = cycle_dataloader(dataloader, num_cycles=1)

    assert next(data_generator).tolist() == list(iter(sampler))[:4]
    assert dataloader.sampler.epoch == 3


def test_cycle_dataloader_advances_from_restored_sampler_epoch():
    dataset = list(range(8))
    sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=7,
    )
    sampler.set_epoch(2)
    dataloader = StatefulDataLoader(dataset, batch_size=4, sampler=sampler)
    data_generator = cycle_dataloader(dataloader, num_cycles=2)

    list(data_generator)

    assert dataloader.sampler.epoch == 3


def test_recovery_at_epoch_boundary_does_not_skip_next_epoch():
    dataset = list(range(8))
    source_sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=7,
    )
    source_dataloader = StatefulDataLoader(
        dataset,
        batch_size=4,
        sampler=source_sampler,
    )
    source_generator = cycle_dataloader(source_dataloader)
    for _ in range(len(source_dataloader)):
        next(source_generator)
    dataloader_state = source_dataloader.state_dict()

    resumed_sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=7,
    )
    resumed_dataloader = StatefulDataLoader(
        dataset,
        batch_size=4,
        sampler=resumed_sampler,
    )
    resumed_dataloader.load_state_dict(dataloader_state)
    recover_info = RecoverInfo(
        last_step_info=StepInfo(
            epoch=0,
            epoch_step=1,
            global_step=1,
            steps_per_epoch=2,
        ),
        saver_info={},
        evaluator_info={},
        stats_logger_info={},
        dataloader_info=dataloader_state,
        checkpoint_info={},
    )
    _restore_sampler_epoch_for_recovery(resumed_dataloader, recover_info)

    expected_sampler = DistributedSampler(
        dataset,
        num_replicas=1,
        rank=0,
        shuffle=True,
        seed=7,
    )
    expected_sampler.set_epoch(1)
    expected_first_batch = list(iter(expected_sampler))[:4]

    resumed_generator = cycle_dataloader(resumed_dataloader)

    assert next(resumed_generator).tolist() == expected_first_batch
    assert resumed_dataloader.sampler.epoch == 1
