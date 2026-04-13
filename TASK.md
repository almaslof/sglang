Stable working version commit is 7c35342c1
HEAD version error is still: "Memory access fault by GPU node-8 (Agent handle: 0x5589f79dc0f0) on address 0x7f2775800000. Reason: Unknown."
Fix error and make diff to working version as small as possible while optimizing performance by pre allocating topk_indices_buffer in the constructor of Indexer.
