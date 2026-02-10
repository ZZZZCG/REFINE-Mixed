# REFINE-Mixed Code

This directory contains a lightweight demo implementation of **REFINE-Mixed**,  
the mixed-source-aware architecture recovery approach proposed in the paper:

> *Enhancing Architecture Recovery for Mixed-Source Software*

- `fusing.py`: fuses 4 embeddings (`functional/dependency/path/license`) into `fused_embeddings.json`.
- `penalty_clustering.py`: adaptive-penalty clustering using `fused_embeddings.json` + `licenses.csv`  + `edges.csv` to output final clusters/components.
