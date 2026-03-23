                                                                            
  ---                                                                       
  Here's a full summary of everything created:                              
                                                                            
  Files created (22 total)                                  

  Model configs — exact ESM-2 parameter match (Table S3)                    
  
  ┌───────────────────────┬────────┬────────┬───────┬──────┬──────────┐     
  │         File          │ Layers │ Hidden │ Heads │ FFN  │  Params  │
  ├───────────────────────┼────────┼────────┼───────┼──────┼──────────┤     
  │ plant_bert_8m.yaml    │ 6      │ 320    │ 20    │ 1280 │ ~8M      │
  ├───────────────────────┼────────┼────────┼───────┼──────┼──────────┤
  │ plant_bert_35m.yaml   │ 12     │ 480    │ 20    │ 1920 │ ~35M     │     
  ├───────────────────────┼────────┼────────┼───────┼──────┼──────────┤
  │ plant_bert_150m.yaml  │ 30     │ 640    │ 20    │ 2560 │ ~150M    │     
  ├───────────────────────┼────────┼────────┼───────┼──────┼──────────┤     
  │ esm2_8m/35m/150m.yaml │ —      │ —      │ —     │ —    │ loads HF │
  └───────────────────────┴────────┴────────┴───────┴──────┴──────────┘     
                                                            
  Training configs — ESM-2 optimizer exactly (β2=0.98, LR=4e-4, WD=0.01, no 
  dropout)                                                  
                                                                            
  - pretrain_8m.yaml → 200K steps, batch=32, accum=4                        
  - pretrain_35m.yaml → 150K steps, batch=16, accum=8
  - pretrain_150m.yaml → 100K steps, batch=8, accum=16                      
  - finetune_plant_bert.yaml + finetune_esm2.yaml           
                                                                            
  Experiment configs (9) — one-liner runs                                   
                                                                            
  python scripts/pretrain.py +experiment=pretrain_plant_8m                  
  python scripts/pretrain.py +experiment=pretrain_plant_35m                 
  python scripts/pretrain.py +experiment=pretrain_plant_150m                
  python scripts/finetune.py +experiment=finetune_plant_bert_8m             
  training.pretrained_checkpoint=<ckpt>                                     
  python scripts/finetune.py +experiment=finetune_esm2_8m          # 
  zero-shot ESM-2                                                           
                                                            
  Source files                                                              
                                                            
  - src/plant_bert/models/esm2_finetune.py — loads ESM-2 from HuggingFace,  
  mean-pools for embeddings, same interface as PlantProteinBERT
  - src/plant_bert/tokenizer/esm2_tokenizer_wrapper.py — wraps HF           
  EsmTokenizer                                                              
  
  Data script                                                               
                                                            
  - scripts/filter_viridiplantae.py — standalone script: reads 18GB DAT.gz, 
  applies taxonomy filter, writes clean FASTA. Run --stats-only for stats
  without writing output.  




 README sections:
  1. Research hypothesis — the plant underrepresentation argument with real
  numbers from your UniRef50 analysis, the contamination issue, and the
  comparison matrix showing what the 3-way experiment proves
  2. Setup — pip install and conda env recreation
  3. Data — filtering statistics (25.5M → 19.9M), what each output file is
  4. Pipeline — all 4 steps with exact commands, including the fast
  FASTA→HDF5 route
  5. Model architectures — parameter table matched to ESM-2 Table S3,
  optimizer hyperparameters with source citations
  6. Repository structure — every file explained in one sentence, organized
  by layer
  7. Downstream tasks — the 4 planned tasks including the negative control
  rationale

  Repo structure explanation (every file):

  ┌──────────┬───────────────────────────────────────────────────────────┐
  │  Layer   │                         Key files                         │
  ├──────────┼───────────────────────────────────────────────────────────┤
  │ Data     │ uniprot_parser.py → preprocessing.py → dataset.py →       │
  │          │ datamodule.py                                             │
  ├──────────┼───────────────────────────────────────────────────────────┤
  │ Models   │ bert.py (PlantBERT) + esm2_finetune.py (baseline) +       │
  │          │ heads.py                                                  │
  ├──────────┼───────────────────────────────────────────────────────────┤
  │ Training │ pretrain.py (MLM) + finetune.py (downstream)              │
  ├──────────┼───────────────────────────────────────────────────────────┤
  │ Scripts  │ filter_viridiplantae.py → fasta_to_hdf5.py → pretrain.py  │
  │          │ → finetune.py → evaluate.py                               │
  ├──────────┼───────────────────────────────────────────────────────────┤
  │ Configs  │ Named experiment YAMLs compose model + training + data    │
  │          │ configs at runtime                                        │
  └──────────┴───────────────────────────────────────────────────────────┘

✻ Baked for 4m 4s