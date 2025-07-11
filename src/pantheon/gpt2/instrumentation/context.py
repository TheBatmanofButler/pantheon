import torch
import wandb
import os


class ObservabilityContextManager:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
        print("Starting observability recording")
        self.run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity=self.config.wandb_entity,
            # Set the wandb project where this run will be logged.
            project=self.config.wandb_project,
            # Track hyperparameters and run metadata.
            config=self.config.to_dict(),
        )

        return self

    def __exit__(self, *_):
        self.run.finish()

    def log(self, step, content):
        self.run.log(content, step=step)


class MemoryContextManager:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
        print("Starting memory recording")
        torch.cuda.memory._record_memory_history(
            max_entries=100000,
        )

        return self

    def __exit__(self, *_):
        local_rank = int(os.environ["LOCAL_RANK"])
        path = self.config.memory_dump_path + f"_{local_rank}.pkl"
        print(f"Dumping memory snapshot at {path}")

        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)

    def oom_observer(self, device, alloc, device_alloc, device_free):
        print("Saving memory snapshot at OOM")
        local_rank = int(os.environ["LOCAL_RANK"])
        path = self.config.memory_dump_path + f"_{local_rank}.pkl"

        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)


class PerformanceContextManager:
    def __init__(self, config):
        self.config = config
        self.profile_ctx_manager = None

    def __enter__(self):
        print("Starting performance recording")
        self.profile_ctx_manager = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=0,
                warmup=0,
                active=4,
                repeat=1,
            ),
            record_shapes=self.config.record_shapes,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.config.performance_profile_path
            ),
        )
        self.profile_ctx_manager.__enter__()

        return self.profile_ctx_manager

    def __exit__(self, *_):
        self.profile_ctx_manager.__exit__(*_)

        print(f"Saving memory timeline at {self.config.memory_timeline_path}")
        self.profile_ctx_manager.export_memory_timeline(
            self.config.memory_timeline_path,
            device="cuda:0",
        )


class ManagedInstrumentor:
    def __init__(
        self,
        manager,
        context=None,
    ):
        self.manager = manager
        self.context = context
