● Everything is clean. Here's the full picture of what was built:                           
                                                                                            
  ---                                                                                       
  What was created (15 files, zero existing files touched)                                  
                                                                                            
  src/plant_bert/adapt/                                                                     
  ├── __init__.py       exports PlantESM2, ESM2Adapter, TokenBudgetCallback                 
  ├── model.py          PlantESM2 wraps EsmForMaskedLM; same interface as PlantProteinBERT  
  └── trainer.py        ESM2Adapter (LightningModule) + TokenBudgetCallback                 
                                                                                            
  scripts/                                                                                  
  └── adapt_esm2.py     entry point with its own Hydra root (configs/esm2_adapt/)           
                                                                                            
  configs/esm2_adapt/                                                                       
  ├── config.yaml               Hydra root — completely separate from configs/config.yaml   
  ├── data/trembl_plant.yaml    reuses UniProtDataModule + existing sequences.h5            
  ├── model/                                                                                
  │   ├── esm2_8m.yaml          facebook/esm2_t6_8M_UR50D                                   
  │   ├── esm2_35m.yaml         facebook/esm2_t12_35M_UR50D                                 
  │   └── esm2_150m.yaml        facebook/esm2_t30_150M_UR50D + gradient_checkpointing       
  ├── training/                                                                             
  │   ├── adapt_8m.yaml         800M token budget, LR=2e-5, batch=64                        
  │   ├── adapt_35m.yaml        800M token budget, LR=2e-5, batch=32                        
  │   └── adapt_150m.yaml       1.6B token budget, LR=1e-5, batch=16, accum=4             
  └── experiment/                                                                           
      ├── adapt_esm2_8m.yaml    one-command launcher                                   
      ├── adapt_esm2_35m.yaml                                                               
      └── adapt_esm2_150m.yaml  primary experiment                                     
                                                                                            
  Three design decisions worth noting:                                                 
                                                                                            
  1. TokenBudgetCallback — training stops precisely when the token budget is consumed,      
  regardless of batch size. max_steps in the trainer is just a ceiling that never fires   
  under normal conditions.                                                                  
  2. HF export after training — save_hf_checkpoint() writes                            
  checkpoints/esm2_adapt/150m/hf_model/ alongside the Lightning .ckpt. This is loadable with
   EsmForMaskedLM.from_pretrained(path) for downstream fine-tuning without Lightning.
  3. UniProtDataModule reused as-is — EsmTokenizer has the same call signature as your      
  custom tokenizer, so zero changes to the data pipeline.                                   
                                                                                          
  To run: python scripts/adapt_esm2.py +experiment=adapt_esm2_150m                          
                                                                                       
✻ Cooked for 8m 25s                                                                         
                                                   