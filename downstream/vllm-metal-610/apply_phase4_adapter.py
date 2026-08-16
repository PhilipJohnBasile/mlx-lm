from pathlib import Path

path = Path("vllm_metal/v1/model_adapter.py")
text = path.read_text()
old = '''        hidden_states = self._forward_target_hidden_states(
            model,
            input_ids,
            cache=cache,
        )
        logits = self._compute_target_logits(model, hidden_states)
        return TargetModelForwardOutput(
            logits=logits,
            hidden_states=self._flatten_target_hidden_states(hidden_states),
        )
'''
new = '''        # Hardened mlx-lm Qwen native-MTP models expose an explicit
        # return_hidden contract: logits have already passed through the final
        # RMSNorm, while the hidden state is the pre-final-norm value the vendor
        # MTP head was trained to consume.
        if bool(getattr(model, "supports_mtp", False)):
            output = model(input_ids, cache=cache, return_hidden=True)
            if not isinstance(output, tuple) or len(output) != 2:
                raise RuntimeError(
                    "supports_mtp model must return (logits, hidden_states) "
                    "when return_hidden=True"
                )
            logits, hidden_states = output
            return TargetModelForwardOutput(
                logits=self.extract_logits(logits),
                hidden_states=self._flatten_target_hidden_states(hidden_states),
            )

        hidden_states = self._forward_target_hidden_states(
            model,
            input_ids,
            cache=cache,
        )
        logits = self._compute_target_logits(model, hidden_states)
        return TargetModelForwardOutput(
            logits=logits,
            hidden_states=self._flatten_target_hidden_states(hidden_states),
        )
'''
if text.count(old) != 1:
    raise RuntimeError("native-MTP target_forward marker mismatch")
path.write_text(text.replace(old, new, 1))
print("Applied Qwen native-MTP hidden/logit adapter contract.")
