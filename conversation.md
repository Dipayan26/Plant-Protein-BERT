✻ Sautéed for 2m 9s                                                                                    
                                                                                          
❯ so if i want to calculate the batch size or a gpu how to calculate for a model manually, and every   
time training i should not fill the entire gpu giving very large batch size ? how to calculate the     
optimal batch size manually                                                                            
                                                                                                       
● Great question. Here's how to think about it from first principles.                                  
                                                                                       
  ---                                                                                                  
  How GPU Memory is Used During Training
                                                                                                       
  When you train, the GPU holds 4 things simultaneously:                               
                                                                                                       
  Total VRAM = Model Weights + Optimizer States + Activations + Gradients                              
                                                                                                       
  ┌───────────────────┬──────────────────────────────────┬──────────────────────────────────┐          
  │     Component     │           Size formula           │     Example (8M model, FP16)     │          
  ├───────────────────┼──────────────────────────────────┼──────────────────────────────────┤          
  │ Model weights     │ params × 2 bytes (FP16)          │ 8M × 2 = 16 MB                   │
  ├───────────────────┼──────────────────────────────────┼──────────────────────────────────┤          
  │ Gradients         │ same as weights                  │ 16 MB                            │          
  ├───────────────────┼──────────────────────────────────┼──────────────────────────────────┤          
  │ Optimizer (AdamW) │ params × 8 bytes (FP32 m+v+copy) │ 8M × 12 = 96 MB                  │          
  ├───────────────────┼──────────────────────────────────┼──────────────────────────────────┤          
  │ Activations       │ depends on batch                 │ THIS is what batch_size controls │
  └───────────────────┴──────────────────────────────────┴──────────────────────────────────┘          
                                                                                       
  The first 3 are fixed — they don't change with batch size. Only activations scale with batch.        
                                                                                       
  ---                                                                                                  
  How to Calculate Activation Memory                                                   
                                                                                                       
  For a BERT-style transformer, the dominant cost is attention matrices:               
                                                                                                       
  Attention memory = batch_size × num_heads × seq_len × seq_len × num_layers × 2 bytes 
                                                                                                       
  For your 8M model (heads=20, layers=6), at avg seq_len=300, FP16:                                    
  batch=32:  32 × 20 × 300 × 300 × 6 × 2 = 691 MB                                                      
  batch=64:  64 × 20 × 300 × 300 × 6 × 2 = 1.4 GB                                                      
  batch=128: 128 × 20 × 300 × 300 × 6 × 2 = 2.7 GB                                                     
  batch=148: 148 × 20 × 300 × 300 × 6 × 2 = 3.2 GB                                                     
                                                                                                       
  Plus hidden states (roughly 30-40% on top of attention). So total VRAM estimate:                     
                                                                                                       
  Total ≈ fixed_cost + (attention_memory × 1.4)                                                        
                                                                                                       
  For 8M model:                                                                                        
  fixed = 16 + 16 + 96 = 128 MB                                                                        
  batch=32:  128 + 691×1.4 = ~1.1 GB  ← very comfortable                                               
  batch=148: 128 + 3200×1.4 = ~4.6 GB ← still ok but seq_len=300 avg                                   
                                                                                                       
  The dangerous case is when you get a batch with long sequences (seq_len=512):                        
  batch=148, seq=512: 148 × 20 × 512 × 512 × 6 × 2 = 9.4 GB + fixed = ~10 GB → OOM risk                
                                                                                                       
  This is exactly why batch=148 becomes unpredictable — average is fine, but one unlucky batch with    
  long sequences will crash.                                                                           
                                                                                                       
  ---                                                                                                  
  The Rule of Thumb for Optimal Batch Size                                             
                                                                                                       
  Step 1 — Calculate your fixed overhead:
  fixed_MB = (params × 12) / 1,000,000   # AdamW FP32 states dominate                                  
                                                                                       
  Step 2 — Leave headroom for activations:                                                             
  available_for_activations = GPU_VRAM × 0.75 - fixed_MB                                               
  Use 75% not 100% — never fill the GPU completely (fragmentation + PyTorch cache overhead).           
                                                                                                       
  Step 3 — Calculate max safe batch size:                                                              
  max_batch = available_for_activations / (heads × seq_len² × layers × 2 bytes × 1.4)                  
                                                                                                       
  For your three models on RTX 3060 (12 GB) at avg seq_len=300:                                        
                                                                                                       
  ┌───────┬────────────┬───────────┬────────────────┐                                                  
  │ Model │ Fixed cost │ Available │ Max safe batch │                                                  
  ├───────┼────────────┼───────────┼────────────────┤                                                  
  │ 8M    │ ~128 MB    │ ~8.9 GB   │ ~400           │                                  
  ├───────┼────────────┼───────────┼────────────────┤                                                  
  │ 35M   │ ~420 MB    │ ~8.6 GB   │ ~170           │                                                  
  ├───────┼────────────┼───────────┼────────────────┤                                                  
  │ 150M  │ ~1.8 GB    │ ~7.2 GB   │ ~40            │                                                  
  └───────┴────────────┴───────────┴────────────────┘                                                  
                                                                                       
  But those are averages. For worst-case sequences (seq_len=512):                                      
                                                                                       
  ┌───────┬──────────────────────────┐                                                                 
  │ Model │ Max safe batch (seq=512) │                                                 
  ├───────┼──────────────────────────┤                                                                 
  │ 8M    │ ~130                     │                                                 
  ├───────┼──────────────────────────┤                                                                 
  │ 35M   │ ~55                      │                                                                 
  ├───────┼──────────────────────────┤                                                                 
  │ 150M  │ ~13                      │                                                                 
  └───────┴──────────────────────────┘                                                                 
                                                                                       
  ---                                                                                                  
  Should You Fill the GPU?                                                             
                                                                                                       
  No. The target is 60–75% VRAM utilization, not 100%. Here's why:
                                                                                                       
  100% VRAM utilization  →  any variation in sequence length → OOM crash               
                         →  PyTorch CUDA cache can't grow → slowdowns                                  
                         →  no room for validation batch                                               
                                                                                                       
  70% VRAM utilization   →  stable across all sequence lengths                                         
                         →  GPU still runs at full compute                                             
                         →  no crashes, predictable training                                           
                                                                                                       
  GPU compute (CUDA cores) and GPU memory are separate. The GPU runs at full compute efficiency at 50% 
  memory utilization — you're not wasting compute by leaving VRAM free.                                
                                                                                                       
  ---                                                                                                  
  The Practical Way to Find Optimal Batch Size
                                                                                                       
  Instead of manual calculation, use this binary search approach:                      
                                                                                                       
  # Start at a conservative estimate, double until near OOM, then back off 20%                         
                                                                                                       
  # Test batch sizes in a short run (100 steps):                                                       
  python scripts/pretrain.py +experiment=pretrain_plant_8m \                                           
      training.trainer.max_steps=100 data.batch_size=64                                                
                                                                                                       
  # Watch nvidia-smi in another terminal:                                                              
  watch -n 1 nvidia-smi                                                                                
                                                                                                       
  # Keep doubling batch_size until you see ~70-75% VRAM usage                                          
  # That's your optimal batch size                                                                     
                                                                                                       
  The number to watch in nvidia-smi:                                                                   
  | GPU  Temp  Perf  ...  |  9000MiB / 12288MiB  |    95% |                                            
                             ^^^^^^^^^^^^^^^^^^^^                                                      
                             target: ~8000-9000 MiB = 65-75%                                           
                                                                                                       
  ---                                                                                                  
  Summary                                                                                              
                                                                                                       
  Optimal batch size = largest batch where VRAM stays under 75%                        
                       at WORST-CASE sequence length in your dataset                                   
                                                                                                       
  For your project (max_seq=512, RTX 3060 12GB):                                                       
    8M model   → batch_size = 64–96   (safe)                                                           
    35M model  → batch_size = 32–48   (safe)                                                           
    150M model → batch_size = 8–16    (safe)                                                           
                                                                                                       
  Always use accumulate_grad_batches to reach your effective batch target                              
  without filling VRAM.                                                                                
                                                                                                       
  Example for 8M: batch_size=64, accumulate_grad_batches=2 → effective batch=128, VRAM ~3GB, fully     
  stable.                                                                                              
                                                                                                       
✻ Brewed for 41s                                       


======================================================================================================
======================================================================================================

❯ i wanna unserstand the repo first to last where to start @src/plant_bert/training/pretrain.py from   
this ?                                                                                                 
  ⎿  Read src/plant_bert/training/pretrain.py (82 lines)

● Here's the complete map of the repo, read it like a pipeline:

  ---
  The Full Picture — Where Everything Lives

  You have plant protein sequences → train a language model → use it for biology tasks

  ---
  Stage 1 — Raw Data → Clean Data

  Where: scripts/parse_uniprot.py → src/plant_bert/data/uniprot_parser.py

  DATA/PLANT_uniprot_new/.../uniprot_trembl_plants.dat.gz   ← 18GB raw UniProt file
          │
          ▼ scripts/parse_uniprot.py  (run once, never again)
          │   uses → src/plant_bert/data/uniprot_parser.py   ← reads the .dat.gz line by line
          │           src/plant_bert/data/preprocessing.py   ← filters: only Viridiplantae,
          │                                                      length 50-1024, no ambiguous AA
          ▼
  outputs/processed/viridiplantae_clean.fasta               ← 19,951,202 sequences
  outputs/processed/trembl_full/sequences.h5                ← 19,938,415 sequences (HDF5 index)
  outputs/processed/info.md                                  ← summary of what was kept/removed

  You only run this once. The HDF5 file is the final product used for all training.

  ---
  Stage 2 — Config System

  Where: configs/

  configs/config.yaml              ← ROOT: lists default sub-configs
  configs/data/trembl_full.yaml    ← path to HDF5, batch_size, max_length
  configs/tokenizer/amino_acid.yaml← vocab of 30 amino acid tokens
  configs/model/plant_bert_8m.yaml ← architecture: 6 layers, hidden=320
  configs/training/pretrain_8m.yaml← optimizer, LR, steps, precision
  configs/experiment/pretrain_plant_8m.yaml  ← GLUES all the above together

  +experiment=pretrain_plant_8m tells Hydra: "load this experiment file, which overrides the defaults with the right
  model/data/training configs."

  ---
  Stage 3 — Tokenizer

  Where: src/plant_bert/tokenizer/amino_acid_tokenizer.py

  protein sequence:  "MKTAYIAKQRQISFVKSHFSRQ..."
          │
          ▼  AminoAcidTokenizer
          │  vocab: 30 tokens [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4, A=5...
          ▼
  token IDs:         [2, 14, 12, 23, 5, ...]   ← numbers the model reads

  ---
  Stage 4 — Dataset + DataModule

  Where: src/plant_bert/data/dataset.py → src/plant_bert/data/datamodule.py

  sequences.h5  (19.9M sequences on disk)
          │
          ▼  StreamingProteinDataset.__getitem__(idx)
          │  opens h5 file lazily (one per worker), reads sequence[idx]
          │  tokenizes it → returns {"input_ids": [...], "attention_mask": [...]}
          │  NOTE: NO padding here (padding=False) — sequences are different lengths
          │
          ▼  DataCollatorForLanguageModeling  (the collate_fn)
          │  receives a LIST of variable-length sequences in one batch
          │  1. pads all to the longest sequence in this batch (dynamic padding)
          │  2. randomly masks 15% of tokens → sets labels=-100 at non-masked spots
          │
          ▼  DataLoader  (inside ProteinDataModule)
          │  batch_size=32, num_workers=4, shuffles every epoch
          ▼
  {
    "input_ids":      [[2, 14, 4, 23, ...], ...],   ← [MASK] inserted
    "attention_mask": [[1,  1, 1,  1, ...], ...],   ← 1=real token, 0=padding
    "labels":         [[−100, −100, 23, −100, ...]]  ← only masked positions have true label
  }

  ---
  Stage 5 — The Model

  Where: src/plant_bert/models/bert.py

  {input_ids, attention_mask, labels}
          │
          ▼  PlantProteinBERT  (wraps HuggingFace BertForMaskedLM)
          │
          │  Embedding layer: token IDs → 320-dim vectors
          │      ↓
          │  Layer 1: Self-Attention → "which amino acids relate to which?"
          │      ↓        Feed-Forward → "process each position"
          │  Layer 2 ... (×6 total for 8M)
          │      ↓
          │  MLM Head: 320-dim → 30-dim (one score per vocab token)
          │
          ▼
  outputs.loss   ← cross-entropy at masked positions (what we minimize)
  outputs.logits ← raw scores for every position (not used during pretraining)

  ---
  Stage 6 — Training Loop

  Where: src/plant_bert/training/pretrain.py ← the file you opened

  MLMPretrainer (LightningModule)
  │
  ├── training_step()     ← called for every batch
  │     model(batch) → loss → log "train/mlm_loss" → return loss
  │     Lightning handles: loss.backward(), optimizer.step(), zero_grad()
  │
  ├── validation_step()   ← called every 5000 steps on held-out data
  │     model(batch) → loss → perplexity=exp(loss) → log both
  │
  └── configure_optimizers()
        AdamW (lr=4e-4, β2=0.98) + linear warmup 2000 steps → linear decay

  ---
  Stage 7 — The Script That Runs Everything

  Where: scripts/pretrain.py

  # This is what YOU run:
  python scripts/pretrain.py +experiment=pretrain_plant_8m

  # It does:
  tokenizer   = AminoAcidTokenizer()
  model       = PlantProteinBERT(6 layers, hidden=320, ...)
  collator    = DataCollatorForLanguageModeling(mlm=True, mlm_probability=0.15)
  datamodule  = ProteinDataModule(hdf5_path=sequences.h5, collate_fn=collator)
  trainer_mod = MLMPretrainer(model, tokenizer, cfg)

  pl.Trainer(max_steps=200000, precision=16-mixed, ...).fit(trainer_mod, datamodule)
                                                      ↑
                                calls training_step() 200,000 times

  ---
  Stage 8 — After Pretraining

  Where: scripts/finetune.py → src/plant_bert/training/finetune.py

  checkpoint: outputs/2026-01-01/12-00-00/checkpoints/pretrain/last.ckpt
          │
          ▼  FineTuner
          │  loads pretrained PlantProteinBERT weights
          │  adds SequenceClassificationHead on top of [CLS] token
          │  trains on labelled plant protein task (GO terms, localization, etc.)
          ▼
  fine-tuned model → evaluate vs ESM-2 on same task

  ---
  The Reading Order

  If you want to understand the code top to bottom:

  1. configs/experiment/pretrain_plant_8m.yaml   ← what is being run
  2. configs/model/plant_bert_8m.yaml            ← what model
  3. src/plant_bert/tokenizer/amino_acid_tokenizer.py
  4. src/plant_bert/data/dataset.py
  5. src/plant_bert/data/datamodule.py
  6. src/plant_bert/models/bert.py
  7. src/plant_bert/training/pretrain.py         ← you are here
  8. scripts/pretrain.py                         ← the entry point

  pretrain.py (training module) is step 7 of 8 — it sits just above the final script that launches everything.

✻ Crunched for 50s



======================================================================================================
======================================================================================================
======================================================================================================