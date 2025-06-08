import torch
import wandb


class ObservabilityContextManager:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
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

    def log(self, content, step):
        self.run.log(
            data=content,
            step=step,
        )


class MemoryContextManager:
    def __init__(self, config):
        self.config = config

    def __enter__(self):
        torch.cuda.memory._record_memory_history()

        return self

    def __exit__(self, *_):
        torch.cuda.memory._dump_snapshot(self.config.memory_dump_path)
        torch.cuda.memory._record_memory_history(enabled=None)


class PerformanceContextManager:
    def __init__(self, config):
        self.config = config
        self.profile_ctx_manager = None

    def __enter__(self):
        self.profile_ctx_manager = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=0, active=5),
            record_shapes=True,
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
        self.profile_ctx_manager.export_memory_timeline(
            self.config.memory_timeline_path,
            device="cuda:0",
        )

    def step(self):
        self.profile_ctx_manager.step()


class ManagedInstrumentor:
    def __init__(
        self,
        manager,
        context=None,
    ):
        self.manager = manager
        self.context = context
