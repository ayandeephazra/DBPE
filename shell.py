import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoTokenizer

#1. Worker Loop
def _worker_loop(local_tok):
    while True:
        cmd = [None]
        dist.broadcast_object_list(cmd, src=0)
        if cmd[0] == "STOP":
            break

        local_docs = [None]
        dist.scatter_object_list(local_docs, None, src=0)

        # Unpack local_docs[0] because scatter_object_list puts the chunk at index 0
        local_ids = [
            local_tok.encode(doc, add_special_tokens=False)
            for doc in local_docs[0]
        ]

        # check to see if worker ranks are being engaged
        local_ids[-1][-1] = str(dist.get_rank())

        dist.gather_object(local_ids, None, dst=0)

# 2. Encoder for Rank 0
class MpiEncoder:
    def __init__(self, local_tok, world_size: int):
        self.local_tok = local_tok
        self.world_size = world_size

    def encode_batch_ids(self, docs: list[str]) -> list[list[int]]:
        dist.broadcast_object_list(["WORK"], src=0)

        chunks = [docs[i::self.world_size] for i in range(self.world_size)]
        local_docs = [None]
        dist.scatter_object_list(local_docs, chunks, src=0)

        # Unpack local_docs[0]
        local_ids = [
            self.local_tok.encode(doc, add_special_tokens=False)
            for doc in local_docs[0]
        ]

        gathered = [None for _ in range(self.world_size)]
        dist.gather_object(local_ids, gathered, dst=0)

        results = [None] * len(docs)
        for r, chunk_ids in enumerate(gathered):
            for idx, ids in enumerate(chunk_ids):
                orig_idx = r + idx * self.world_size
                results[orig_idx] = ids
        return results

    def close(self):
        dist.broadcast_object_list(["STOP"], src=0)


# 3. THE ENTRY POINT FOR EVERY PROCESS
def run(rank: int, world_size: int, model_name: str):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # Every process loads its own copy of the tokenizer
    local_tok = AutoTokenizer.from_pretrained(model_name)

    if rank == 0:
        # Rank 0 acts as the manager and runs your main test code
        encoder = MpiEncoder(local_tok, world_size)

        prompts = [
            "Hello world!",
            "Distributed tokenization with MPI.",
            "Rank 0 splits the work.",
            "Workers tokenize in parallel.",
            "Results come back in order.",
        ]
        token_ids = encoder.encode_batch_ids(prompts)

        for text, ids in zip(prompts, token_ids):
            print(f"{text!r} -> {ids}")

        # Shut down the workers cleanly before destroying the process group
        encoder.close()
    else:
        # Ranks 1, 2, 3... sit in the worker loop until encoder.close() is called
        _worker_loop(local_tok)

    dist.destroy_process_group()


# 4. MAIN LAUNCHER
def main():
    world_size = 4
    model_name = "gpt2"  # Small, fast tokenizer for testing without gated access
    mp.spawn(run, args=(world_size, model_name), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()

