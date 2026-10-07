"""One-GPU verl FSDP engine bridge for the bounded RL/OPD pilot.

The engine owns model construction, backward, clipping, AdamW, and checkpointing.
This adapter supplies the frozen experiment's combined clipped objective and HF
rollouts. It is not the stock Ray trainer or a vLLM rollout implementation.
Pinned upstream: 8718ca30a3f002f93b7c4fd99b9b2506718681bc.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile

import torch

VERL_COMMIT = "8718ca30a3f002f93b7c4fd99b9b2506718681bc"


def tree_fingerprint(value):
    """Stable exact hash of tensor bytes and scalar/container state."""
    import json
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            h.update(f"tensor:{item.dtype}:{tuple(item.shape)}".encode())
            raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            h.update(memoryview(raw))
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=lambda x: (type(x).__name__, repr(x))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            h.update((type(item).__name__ + ":" + json.dumps(item, allow_nan=False)).encode())
    visit(value)
    return h.hexdigest()


def clipped_response_loss(current, old, signals, mask, response_weights):
    """Token mean per response; explicit weights preserve group/response means.

    Padding is excluded before exp(), so even arbitrary padded values cannot
    affect the loss or its gradient. EOS is an ordinary included response token.
    """
    assert current.shape == old.shape == signals.shape == mask.shape
    mask = mask.bool()
    assert mask.any(dim=1).all()
    assert response_weights.shape == (current.shape[0],)
    assert not old.requires_grad and not signals.requires_grad
    assert torch.isfinite(current[mask]).all()
    assert torch.isfinite(old[mask]).all() and torch.isfinite(signals[mask]).all()
    delta = torch.where(mask, current.float() - old.detach().float(), 0.0)
    advantage = torch.where(mask, signals.detach().float(), 0.0)
    ratio = delta.exp()
    per_token = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
    per_response = (per_token * mask).sum(1) / mask.sum(1)
    return (per_response * response_weights).sum()


def make_batch(groups, pad_token_id, old=None, signals=None):
    """Convert frozen rollout records to verl's jagged input contract on CPU."""
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding

    assert groups and all(group["answers"] for group in groups)
    records = [(g, a, gi, ai) for gi, g in enumerate(groups)
               for ai, a in enumerate(g["answers"])]
    n = len(records)
    pwidth = max(len(g["prompt_token_ids"]) for g in groups)
    rwidth = max(len(a["token_ids"]) for _, a, _, _ in records)
    assert pwidth > 0 and rwidth > 0
    prompts = torch.full((n, pwidth), pad_token_id, dtype=torch.long)
    responses = torch.full((n, rwidth), pad_token_id, dtype=torch.long)
    attention = torch.zeros((n, pwidth + rwidth), dtype=torch.long)
    mask = torch.zeros((n, rwidth), dtype=torch.bool)
    previous = torch.zeros((n, rwidth), dtype=torch.float32)
    advantages = torch.zeros_like(previous)
    weights = torch.empty(n, dtype=torch.float32)
    for i, (group, answer, gi, ai) in enumerate(records):
        prompt, completion = group["prompt_token_ids"], answer["token_ids"]
        assert prompt and completion
        plen, rlen = len(prompt), len(completion)
        prompts[i, -plen:] = torch.tensor(prompt)
        responses[i, :rlen] = torch.tensor(completion)
        attention[i, pwidth - plen:pwidth + rlen] = 1
        mask[i, :rlen] = True
        weights[i] = 1.0 / (len(groups) * len(group["answers"]))
        if old is not None:
            assert old[gi][ai].shape == (rlen,) and not old[gi][ai].requires_grad
            previous[i, :rlen] = old[gi][ai].detach().float().cpu()
        if signals is not None:
            assert signals[gi][ai].shape == (rlen,) and not signals[gi][ai].requires_grad
            advantages[i, :rlen] = signals[gi][ai].detach().float().cpu()
    positions = (attention.cumsum(-1) - 1).clamp_min(0) * attention
    data = TensorDict({
        "input_ids": torch.cat((prompts, responses), 1),
        "prompts": prompts, "responses": responses,
        "attention_mask": attention, "position_ids": positions,
        "response_mask": mask, "old_log_probs": previous,
        "effective_advantages": advantages, "response_weights": weights,
    }, batch_size=[n])
    data = left_right_2_no_padding(data)
    tu.assign_non_tensor(data, temperature=1.0, use_remove_padding=False,
                        use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
                        use_fused_kernels=False, pad_token_id=pad_token_id,
                        global_batch_size=n, calculate_entropy=False)
    return data


def experiment_loss(model_output, data, dp_group=None, enforce_old_match=True):
    from verl.workers.utils.padding import no_padding_2_padding
    current = no_padding_2_padding(model_output["log_probs"], data).float()
    old, signals = data["old_log_probs"], data["effective_advantages"]
    mask = data["response_mask"].bool()
    difference = float((current.detach()[mask] - old[mask]).abs().max())
    if enforce_old_match:
        assert difference <= 1e-3, f"Pre-update policy mismatch: {difference}"
    loss = clipped_response_loss(current, old, signals, mask, data["response_weights"])
    assert torch.isfinite(loss)
    return loss, {"max_preupdate_logp_difference": difference,
                  "backward_responses": int(current.shape[0])}


class VerlEngineBridge:
    def __init__(self, config, initial_adapter_path, backend="fsdp"):
        from verl.trainer.config import CheckpointConfig
        from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

        assert backend == "fsdp", "This audited bridge is for the one-rank FSDP1 pilot."
        assert torch.cuda.is_available()
        assert config["lora"]["lora_dropout"] == 0.0
        os.environ["VERL_DISABLE_FLASH_ATTN_CE"] = "1"
        torch.cuda.set_device(0)
        self._owns_process_group = not torch.distributed.is_initialized()
        self._rendezvous_dir = None
        if self._owns_process_group:
            self._rendezvous_dir = tempfile.TemporaryDirectory(prefix="verl-pilot-nccl-")
            rendezvous = Path(self._rendezvous_dir.name) / "rendezvous"
            torch.distributed.init_process_group(
                backend="nccl", init_method=rendezvous.as_uri(), rank=0, world_size=1)
        assert torch.distributed.get_world_size() == 1
        self.config = config
        self.call_counts = {"train_batch": 0, "infer_batch": 0, "generate": 0,
                            "save_checkpoint": 0, "load_checkpoint": 0}
        self.initial_adapter_path = str(Path(initial_adapter_path).resolve())

        class FP32LogProbabilityEngine(FSDPEngineWithLMHead):
            # The reference computes CE from FP32 logits. Upstream's ordinary
            # BF16 fallback computes BF16 log_softmax, which changes the signal.
            def prepare_model_outputs(inner, output, output_args, micro_batch, logits_processor_func):
                output.logits = output.logits.float()
                return super().prepare_model_outputs(
                    output, output_args, micro_batch, logits_processor_func)

        lora = config["lora"]
        self.model_config = HFModelConfig(
            path=config["models"]["student"]["local_snapshot"],
            override_config={"attn_implementation": "sdpa"},
            trust_remote_code=False, enable_gradient_checkpointing=False,
            enable_activation_offload=False, use_remove_padding=False,
            use_fused_kernels=False, use_liger=False, use_shm=False,
            lora_rank=lora["r"], lora_alpha=lora["lora_alpha"],
            target_modules=lora["target_modules"], lora_adapter_path=self.initial_adapter_path,
        )
        engine_config = FSDPEngineConfig(
            strategy=backend, fsdp_size=1, model_dtype="bf16", use_orig_params=False,
            mixed_precision={"param_dtype": "bf16", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
            use_torch_compile=False, use_remove_padding=False, use_fused_kernels=False,
            use_dynamic_bsz=False, micro_batch_size_per_gpu=1,
            infer_micro_batch_size_per_gpu=1, seed=config["seed"],
            param_offload=False, optimizer_offload=False, forward_prefetch=False,
        )
        optimizer_config = FSDPOptimizerConfig(
            lr=config["learning_rate"], weight_decay=0.0, betas=(0.9, 0.999),
            clip_grad=1.0, total_training_steps=8, lr_warmup_steps=0,
            lr_warmup_steps_ratio=0.0, lr_scheduler_type="constant",
            optimizer="AdamW", optimizer_impl="torch.optim",
            override_optimizer_config={"eps": 1e-8},
        )
        checkpoint_config = CheckpointConfig(
            save_contents=["model", "optimizer", "extra"],
            load_contents=["model", "optimizer", "extra"], save_lora_only=True)
        self.engine = FP32LogProbabilityEngine(
            self.model_config, engine_config, optimizer_config, checkpoint_config)
        self.engine.initialize()
        self.model = self.engine.module
        self.model.eval()
        self.model.config.use_cache = False
        self.tokenizer = self.model_config.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        assert named and all("lora_A" in n or "lora_B" in n for n, _ in named)
        assert all(p.dtype == torch.float32 for _, p in named), "LoRA master parameters must remain FP32."
        self.trainable_parameters = sum(p.numel() for _, p in named)
        assert self.trainable_parameters == 17432576

    @property
    def device(self):
        return torch.device("cuda:0")

    @property
    def optimizer(self):
        return self.engine.optimizer

    def score_groups(self, groups):
        from verl.workers.utils.padding import no_padding_2_padding
        data = make_batch(groups, self.tokenizer.pad_token_id)
        with self.engine.eval_mode():
            result = self.engine.infer_batch(data)
            self.call_counts["infer_batch"] += 1
        # Engine returns GPU outputs, while original batch metadata is on CPU.
        padded = no_padding_2_padding(result["model_output"]["log_probs"], data.to(self.device))
        assert padded.dtype == torch.float32 and torch.isfinite(padded).all()
        values, index = [], 0
        for group in groups:
            current = []
            for answer in group["answers"]:
                current.append(padded[index, :len(answer["token_ids"])].detach().clone())
                index += 1
            values.append(current)
        return values

    def update(self, groups, old, signals):
        data = make_batch(groups, self.tokenizer.pad_token_id, old, signals)
        if not torch.count_nonzero(data["effective_advantages"]):
            self.engine.optimizer_zero_grad()
            return {"optimizer_steps": 0, "skip_reason": "all_effective_signals_zero",
                    "loss_before": 0.0, "gradient_norm_before_clipping": 0.0,
                    "max_preupdate_logp_difference": 0.0, "backward_responses": 0}
        with self.engine.train_mode():
            # Keep the reference's deterministic dropout-free forward behavior.
            self.model.eval()
            result = self.engine.train_batch(data, experiment_loss)
            self.call_counts["train_batch"] += 1
        norm = float(result["metrics"]["grad_norm"])
        assert math.isfinite(norm) and norm > 0, "Native engine produced an invalid/zero gradient norm."
        assert all(torch.isfinite(p).all() for p in self.model.parameters() if p.requires_grad)
        assert all(p.grad is None for p in self.model.parameters() if not p.requires_grad)
        self.engine.lr_scheduler_step()
        self.engine.optimizer_zero_grad()
        return {"optimizer_steps": 1, "skip_reason": None,
                "loss_before": sum(result["loss"]),
                "gradient_norm_before_clipping": norm,
                "max_preupdate_logp_difference": max(result["metrics"]["max_preupdate_logp_difference"]),
                "backward_responses": sum(result["metrics"]["backward_responses"])}

    def generate(self, **kwargs):
        """HF rollout from the current live adapter, following upstream FSDP context."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # Transformers 5.12 removed this older generate argument. Every active
        # sampling control is explicit, so removing it cannot restore defaults.
        kwargs.pop("use_model_defaults", None)
        generation = kwargs.get("generation_config")
        assert generation is not None, "Pass the frozen explicit GenerationConfig."
        resolved, unused = self.model._prepare_generation_config(
            generation, **{k: v for k, v in kwargs.items()
                           if k not in ("generation_config", "input_ids", "attention_mask")})
        if resolved.do_sample:
            for key, expected in {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0}.items():
                assert getattr(resolved, key) == expected, f"Unexpected sampling default: {key}"
        with self.engine.eval_mode(), torch.inference_mode():
            with FSDP.summon_full_params(self.model, writeback=False, recurse=False):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = self.model.generate(**kwargs)
                    self.call_counts["generate"] += 1
                    return result

    def adapter_snapshot(self):
        from peft import get_peft_model_state_dict
        state = self.model.state_dict()
        adapter = get_peft_model_state_dict(self.model, state_dict=state, save_embedding_layers=False)
        return {name: tensor.detach().cpu().clone() for name, tensor in adapter.items()}

    def base_fingerprint(self):
        h = hashlib.sha256()
        for name, tensor in self.model.state_dict().items():
            if "lora_A." in name or "lora_B." in name:
                continue
            h.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
            h.update(memoryview(tensor.detach().contiguous().cpu().view(torch.uint8).numpy()))
        return h.hexdigest()

    def export_adapter(self, path):
        underlying = self.model._fsdp_wrapped_module
        underlying.save_pretrained(str(path), state_dict=self.model.state_dict(),
                                   safe_serialization=True, save_embedding_layers=False)

    def save_checkpoint(self, path, step):
        # Atomic publication/manifest and pruning belong to the controller.
        self.engine.save_checkpoint(str(path), global_step=step, max_ckpt_to_keep=None)
        self.call_counts["save_checkpoint"] += 1

    def load_checkpoint(self, path):
        self.engine.load_checkpoint(str(path), del_local_after_load=False)
        self.call_counts["load_checkpoint"] += 1

    def framework_metadata(self):
        import inspect
        from dataclasses import asdict
        return {
            "verl_commit": VERL_COMMIT,
            "engine_class": type(self.engine).__qualname__,
            "upstream_engine_class": type(self.engine).__bases__[0].__module__ + "." + type(self.engine).__bases__[0].__name__,
            "train_batch_implementation": self.engine.train_batch.__func__.__module__ + "." + self.engine.train_batch.__func__.__qualname__,
            "train_batch_source": inspect.getsourcefile(self.engine.train_batch),
            "optimizer": type(self.optimizer).__module__ + "." + type(self.optimizer).__name__,
            "call_counts_this_process": dict(self.call_counts),
            "base_storage": "bfloat16", "adapter_master_storage": "float32",
            "forward_autocast": "bfloat16", "selected_log_probability_dtype": "float32",
            "precision_extension": "cast output logits to FP32 before native engine logprob calculation",
            "loss_reduction": "token mean per response, response mean per question, question mean",
            "native_microbatch_responses": 1, "checkpoint_lora_only": True,
            "effective_engine_config": asdict(self.engine.engine_config),
            "effective_optimizer_config": asdict(self.engine.optimizer_config),
            "effective_model_config": {key: getattr(self.model_config, key) for key in
                ("path", "override_config", "enable_gradient_checkpointing", "use_remove_padding",
                 "use_fused_kernels", "use_liger", "lora_rank", "lora_alpha", "target_modules", "lora_adapter_path")},
            "checkpoint_contents": ["model", "optimizer", "extra"],
            "rollout_backend": "HF generate with FSDP full-parameter context; no vLLM",
        }

    def optimizer_fingerprint(self):
        return tree_fingerprint({"optimizer": self.optimizer.state_dict(),
                                 "scheduler": self.engine.lr_scheduler.state_dict()})

    def close(self):
        if self._owns_process_group and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if self._rendezvous_dir is not None:
            self._rendezvous_dir.cleanup()


def cpu_parity_checks():
    """Independent formula/gradient and real verl padding conversion checks."""
    from verl.workers.utils.padding import no_padding_2_padding
    old = torch.tensor([[-2., -3., -4., 0.], [-1., -2., 0., 0.]])
    current = (old + torch.tensor([[math.log(1.4), math.log(0.6), math.log(1.1), 99.],
                                  [math.log(1.4), math.log(0.6), 99., 99.]])).requires_grad_()
    signals = torch.tensor([[1., -2., -1., 123.], [-1., 2., 321., 432.]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    weights = torch.tensor([0.5, 0.5])
    actual = clipped_response_loss(current, old, signals, mask, weights)
    independent = current.detach().clone().requires_grad_()
    reference = 0
    for i, length in enumerate([3, 2]):
        ratio = (independent[i, :length] - old[i, :length]).exp()
        a = signals[i, :length]
        reference = reference - torch.minimum(ratio * a, torch.clamp(ratio, .8, 1.2) * a).mean() / 2
    grad_actual = torch.autograd.grad(actual, current)[0]
    grad_reference = torch.autograd.grad(reference, independent)[0]
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(grad_actual, grad_reference, rtol=0, atol=0)
    assert not grad_actual[~mask].count_nonzero()
    groups = [{"prompt_token_ids": [10, 11], "answers": [{"token_ids": [20, 21, 99]}]},
              {"prompt_token_ids": [12], "answers": [{"token_ids": [22, 99]}]}]
    batch = make_batch(groups, 0)
    raw = torch.nested.as_nested_tensor([torch.arange(5.).requires_grad_(),
                                         torch.arange(3.).requires_grad_()], layout=torch.jagged)
    selected = no_padding_2_padding(raw, batch)
    torch.testing.assert_close(selected, torch.tensor([[1., 2., 3.], [0., 1., 0.]]))
    assert batch["response_mask"].tolist() == [[True, True, True], [True, True, False]]
    selected.sum().backward()
    class FakeEngine:
        def __init__(self):
            self.zero_calls = 0
        def optimizer_zero_grad(self):
            self.zero_calls += 1
        def train_batch(self, *args):
            raise AssertionError("Zero-signal batch must not reach engine AdamW.")
    bridge = object.__new__(VerlEngineBridge)
    bridge.engine = FakeEngine()
    bridge.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    scores = [[torch.zeros(3)], [torch.zeros(2)]]
    skipped = bridge.update(groups, scores, scores)
    assert skipped["optimizer_steps"] == 0 and bridge.engine.zero_calls == 1
    return {"status": "pass", "loss_gradient_parity": True, "positive_negative_clipping": True,
            "unequal_response_lengths": True, "prompt_padding_excluded_eos_included": True,
            "all_zero_skips_native_optimizer": True, "test_loss": float(actual.detach())}


if __name__ == "__main__":
    import json
    print(json.dumps(cpu_parity_checks(), indent=2))
