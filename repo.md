                                                                            
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