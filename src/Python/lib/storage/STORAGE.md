# Storage Program Flow & Control

```mermaid
classDiagram
    class HashFile {
        +Path path
        +int byte_chunk_size
        +read_hash_from_file() str
        +write_hash_to_file(hash_value, overwrite_hashfile)
        +verify_file() bool
        +delete()
        +__hash__() str
    }

    class LockFile {
        +Path path
        +bool is_locked
        +acquire_lock()
        +release_lock(ignore_lock)
        +verify_lock_file() bool
        +refresh()
    }

    class MetadataFile {
        +Path path
        +LockFile lock
        +HashFile hash
        +read(ignore_lock) dict
        +write(metadata, overwrite)
    }

    class ShardFile {
        +Path path
        +read() NumpyIntegerArray
        +write(chunk_dict, overwrite)
        +delete()
        +validate(chunk_dict)
    }

    class PartitionPlan {
        <<Dataclass>>
        +Integer items_per_shard
        +Integer chunks_per_shard
        +Integer items_per_chunk
        +Integer total_shards
        +Integer total_chunks
    }

    class PartitionStrategy {
        +calculate_plan(total_items, item_byte_size, ...) PartitionPlan$
    }

    class ShardManager {
        +save(primes_array, ...)
        +load(min, max, ...) NumpyIntegerArray
        +repartition_shards(...)
        +verify_shard_integrity() bool
    }

    MetadataFile *-- LockFile
    MetadataFile *-- HashFile
    
    ShardManager ..> MetadataFile : uses
    ShardManager ..> ShardFile : uses
    ShardManager ..> PartitionStrategy : uses
    PartitionStrategy ..> PartitionPlan : creates
   SM->>SM: Check interval intersection
   opt Intersects
       SM->>SF: __init__(shard_path)
       activate SF
       SM->>SF: read()
```   SM->>SM: _partitions(...)
   SM->>SF: __init__(path)
   activate SF
   SM->>SF: write(shard_dict, overwrite)
   SF-->>SM: void
   deactivate SF
