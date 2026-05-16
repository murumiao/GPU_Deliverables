# How to build
run ```make all```

# How to download matrixes
run ```./scripts/download_matrixes.sh```. This script will download into ```./data``` 10 matrixes from SuiteSparse and sort them.

If you have matrixes and they are not sorted, please run ```./scripts/sort_matrixes.sh```

# How to run
By default, the binaries are located in the folder `./bin` 
### CPU implementations
```<path_to_script> <path_to_matrix>```


### GPU implementations
<b>COO implementation:</b>

```
<path_to_script> <path_to_matrix> <mode[0,1]> <n_blocks> <n_threads_per_block>
```

- `mode=0` for sequential access;
- `mode=1` for strided access.

<b>CSR implementations</b>
```
<path_to_script> <path_to_matrix> <mode[0,1,2,3]> <n_threads_per_block> <shared_mem_size>
```
- `mode=0` for sequential access and global memory;
- `mode=1` for strided access and global memory;
- `mode=2` for strided access with shared memory;
- `mode=3` for strided access with shared memeory and improvements on bank conflicts.
