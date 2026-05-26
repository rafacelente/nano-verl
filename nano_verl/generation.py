import torch

from vllm import LLM, SamplingParams


class RolloutEngine:
    def __init__(
        self,
        model_name: str,
        max_response_length: int = 4096,
    ):
        self.llm = LLM(
            model=model_name,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )

        self.max_response_length = max_response_length

        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_response_length,
            logprobs=0,
        )

    def generate(self, prompt_token_ids_list: list[list[int]]):
        outputs = self.llm.generate(
            prompts=prompt_token_ids_list,
            sampling_params=self.sampling_params,
        )

        response_ids = []
        old_log_probs = []

        for output in outputs:
            tokens = list(output.outputs[0].token_ids)
            response_ids.append(tokens)

            lps = [
                logprobs[tokens[i]].logprob for i, logprobs in enumerate(output.outputs[0].logprobs)
            ]
            old_log_probs.append(lps)
        
        return {"response_ids": response_ids, "old_log_probs": old_log_probs}

    def update_weights(self, named_parameters: torch.nn.Parameter):
        weights = [(name, param.data) for name, param in named_parameters]

        self.llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(weights)