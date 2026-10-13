from pathlib import Path

p = Path(r".\ASR\manipuri_asr_v2\train_b2_partial_encoder.py")
s = p.read_text(encoding="utf-8")

# ------------------------------------------------------------
# 1. Disable gradient checkpointing
# ------------------------------------------------------------

old = '''        try:

            self.encoder.gradient_checkpointing_enable()

            print(
                "[MODEL] Gradient checkpointing: ENABLED"
            )

        except Exception as e:

            print(
                "[MODEL] Gradient checkpointing unavailable:",
                repr(e),
            )
'''

new = '''        # Disabled for B2 Windows/CUDA stability.
        # Only layers 20-23 are trainable, so the additional
        # checkpointing complexity is not necessary for this run.

        try:
            self.encoder.gradient_checkpointing_disable()
        except Exception:
            pass

        print(
            "[MODEL] Gradient checkpointing: DISABLED"
        )
'''

if old not in s:
    raise RuntimeError("Could not find gradient-checkpointing block")

s = s.replace(old, new, 1)

# ------------------------------------------------------------
# 2. Add checkpoint helpers before TRAIN ONE EPOCH
# ------------------------------------------------------------

marker = '''# ============================================================
# TRAIN ONE EPOCH
# ============================================================
'''

helpers = r'''
# ============================================================
# B2 RECOVERY / DEBUG HELPERS
# ============================================================

CHECKPOINT_EVERY_OPT_STEPS = 500


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")

    torch.save(obj, tmp)

    if path.exists():
        path.unlink()

    tmp.replace(path)


def save_recovery_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    batch_step,
    optimizer_step,
    running_loss,
    sample_info,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "epoch": int(epoch),
        "batch_step": int(batch_step),
        "optimizer_step": int(optimizer_step),
        "running_loss": float(running_loss),
        "sample_info": sample_info,
        "timestamp": time.time(),
    }

    path = RUN_DIR / "last_safe.pt"

    atomic_torch_save(
        payload,
        path,
    )

    print(
        f"[RECOVERY] Saved optimizer_step={optimizer_step} "
        f"batch_step={batch_step} -> {path}",
        flush=True,
    )


def describe_batch(batch):
    try:
        return {
            "text": str(batch["texts"][0]),
        }
    except Exception:
        return {
            "text": "<unavailable>",
        }


'''

s = s.replace(marker, helpers + marker, 1)

# ------------------------------------------------------------
# 3. Replace train_one_epoch entirely
# ------------------------------------------------------------

start = s.index("def train_one_epoch(")
end = s.index("\n\n# ============================================================\n# VALIDATION", start)

new_train = r'''def train_one_epoch(
    model,
    processor,
    loader,
    optimizer,
    scaler,
    ctc_loss_fn,
    epoch,
    optimizer_step_start=0,
):
    model.train()

    # Frozen portion stays in eval mode.
    model.encoder.feature_projection.eval()

    for idx in range(20):
        model.encoder.encoder.layers[idx].eval()

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    items = 0
    optimizer_steps = optimizer_step_start

    start_time = time.time()

    for step, batch in enumerate(loader, start=1):

        sample_info = describe_batch(batch)

        try:

            if step % 100 == 1 or step % 100 == 0:
                print(
                    f"[DATA] epoch={epoch} "
                    f"batch={step}/{len(loader)} "
                    f"text={sample_info['text'][:80]}",
                    flush=True,
                )

            input_features = make_input_features(
                processor,
                batch["audio"],
            )

            input_features = input_features.to(
                DEVICE,
                non_blocking=True,
            )

            targets = batch["targets"].to(DEVICE)

            target_lengths = batch["target_lengths"].to(DEVICE)

            use_amp = (
                AMP
                and DEVICE.type == "cuda"
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(
                    input_features
                )

                # Force CUDA errors to surface here.
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                input_lengths = torch.full(
                    (logits.shape[0],),
                    logits.shape[1],
                    dtype=torch.long,
                    device=DEVICE,
                )

                log_probs = (
                    logits
                    .float()
                    .log_softmax(dim=-1)
                    .transpose(0, 1)
                )

                loss = ctc_loss_fn(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths,
                )

                scaled_loss = loss / GRAD_ACCUM

            if scaler is not None:
                scaler.scale(
                    scaled_loss
                ).backward()
            else:
                scaled_loss.backward()

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            should_step = (
                step % GRAD_ACCUM == 0
                or step == len(loader)
            )

            if should_step:

                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                optimizer_steps += 1

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                if (
                    optimizer_steps
                    % CHECKPOINT_EVERY_OPT_STEPS
                    == 0
                ):

                    save_recovery_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        batch_step=step,
                        optimizer_step=optimizer_steps,
                        running_loss=running_loss,
                        sample_info=sample_info,
                    )

            running_loss += (
                float(loss.item())
                * len(batch["texts"])
            )

            items += len(batch["texts"])

            if step % 100 == 0:

                mem = 0.0

                if DEVICE.type == "cuda":
                    mem = (
                        torch.cuda.memory_allocated(DEVICE)
                        / 1024**3
                    )

                print(
                    f"[TRAIN] epoch={epoch} "
                    f"step={step}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"opt_step={optimizer_steps} "
                    f"VRAM={mem:.2f}GB",
                    flush=True,
                )

        except KeyboardInterrupt:
            raise

        except Exception as exc:

            print(
                "\n" + "=" * 72,
                flush=True,
            )

            print(
                "[BATCH ERROR]",
                flush=True,
            )

            print(
                f"epoch={epoch}",
                flush=True,
            )

            print(
                f"batch_step={step}",
                flush=True,
            )

            print(
                f"optimizer_step={optimizer_steps}",
                flush=True,
            )

            print(
                f"sample={sample_info}",
                flush=True,
            )

            print(
                f"exception={type(exc).__name__}: {exc}",
                flush=True,
            )

            print(
                "=" * 72,
                flush=True,
            )

            # Save whatever state is still valid.
            try:
                save_recovery_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    batch_step=step,
                    optimizer_step=optimizer_steps,
                    running_loss=running_loss,
                    sample_info=sample_info,
                )
            except Exception as save_exc:
                print(
                    "[RECOVERY] Save failed:",
                    repr(save_exc),
                    flush=True,
                )

            raise

    return (
        running_loss / max(items, 1),
        time.time() - start_time,
        optimizer_steps,
    )
'''

s = s[:start] + new_train + s[end:]

# ------------------------------------------------------------
# 4. Change main() call for the new return values
# ------------------------------------------------------------

old_call = '''        train_loss, elapsed = (
            train_one_epoch(
                model,
                model.processor,
                train_loader,
                optimizer,
                scaler,
                ctc_loss_fn,
                epoch,
            )
        )
'''

new_call = '''        train_loss, elapsed, optimizer_steps = (
            train_one_epoch(
                model,
                model.processor,
                train_loader,
                optimizer,
                scaler,
                ctc_loss_fn,
                epoch,
                optimizer_step_start=0,
            )
        )
'''

if old_call not in s:
    raise RuntimeError("Could not find train_one_epoch call")

s = s.replace(old_call, new_call, 1)

# ------------------------------------------------------------
# 5. Flush final output
# ------------------------------------------------------------

s = s.replace(
    'print(\n            f"[TRAIN] loss={train_loss:.4f} "\n            f"time={elapsed:.1f}s"\n        )',
    'print(\n            f"[TRAIN] loss={train_loss:.4f} "\n            f"time={elapsed:.1f}s "\n            f"optimizer_steps={optimizer_steps}",\n            flush=True,\n        )',
    1,
)

p.write_text(s, encoding="utf-8")

print("B2 STABILITY PATCH: PASS")
print("Gradient checkpointing: DISABLED")
print("Recovery checkpoint interval: 500 optimizer steps")
print("CUDA synchronization: ENABLED")
print("Batch tracing: ENABLED")
print("Atomic checkpoint writing: ENABLED")
print("Backup remains at:")
print("  .\\ASR\\manipuri_asr_v2\\train_b2_partial_encoder_backup.py")
