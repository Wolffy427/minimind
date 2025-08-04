# lm_eval/models/minimind_lm.py
import torch
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from model.model_minimind import MiniMindForCausalLM
from transformers import AutoTokenizer

@register_model("minimind")
class MiniMindLM(LM):
    def __init__(self, pretrained, device="cuda", **kwargs):
        super().__init__()
        self.device = device
        self.model = MiniMindForCausalLM.from_pretrained(
            pretrained,
            torch_dtype=torch.bfloat16,
            **kwargs
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.vocab_size = self.tokenizer.vocab_size

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return 1  # 可改

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            return self.model(inps).logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context,
            max_length=max_length,
            eos_token_id=eos_token_id,
            do_sample=False
        )

    # 下面三个方法是必须的，用于评估
    def loglikelihood(self, requests):
        from lm_eval.api.instance import Instance
        results = []
        for req in requests:
            context, continuation = req.args
            full = context + continuation
            full_ids = self.tok_encode(full)
            ctx_ids = self.tok_encode(context)
            cont_ids = self.tok_encode(continuation)

            input_ids = torch.tensor([full_ids], device=self.device)
            logits = self._model_call(input_ids)[:, :-1, :]
            cont_len = len(cont_ids)
            logits = logits[:, -cont_len:, :]
            target_ids = torch.tensor(cont_ids, device=self.device).unsqueeze(0)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loglikelihood = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).sum().item()
            greedy_tokens = logits.argmax(dim=-1)
            is_greedy = int((greedy_tokens == target_ids).all().item())
            results.append((loglikelihood, is_greedy))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for req in requests:
            string, = req.args
            tokens = self.tok_encode(string)
            input_ids = torch.tensor(tokens, device=self.device).unsqueeze(0)
            logits = self._model_call(input_ids)[:, :-1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            target_ids = torch.tensor(tokens[1:], device=self.device).unsqueeze(0)
            loglikelihood = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1).sum().item()
            results.append((loglikelihood,))
        return results

    def generate_until(self, requests):
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            context_ids = self.tok_encode(context)
            input_ids = torch.tensor([context_ids], device=self.device)
            gen = self._model_generate(
                input_ids,
                max_length=len(context_ids) + max_gen_toks,
                eos_token_id=self.eot_token_id
            )
            gen_tokens = gen[0][len(context_ids):]
            result = self.tok_decode(gen_tokens)
            for stop_seq in until:
                if stop_seq in result:
                    result = result.split(stop_seq)[0]
            results.append(result)
        return results