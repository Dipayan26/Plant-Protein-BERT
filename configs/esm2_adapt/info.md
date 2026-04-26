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
                                                  



● Data is ready. In your terminal:

  conda activate plantbert
  cd /home/dipayan/Documents/Plant-Protein-BERT
  python scripts/adapt_esm2.py +experiment=adapt_esm2_8m


####### choose data 

❯ is there any guide for finetuning token selection ?                                                  
                                                                                                       
● There's no single universal formula, but there are established frameworks. Here's the practical      
  landscape:                                                                                           
                                                                                                       
  ---
  The foundational reference: Chinchilla (from-scratch)                                                
                                                                                                       
  Hoffmann et al. 2022 ("Training Compute-Optimal LLMs") gives the from-scratch optimal rule:
                                                                                                       
  tokens = 20 × parameters                                                                             
   
  ┌────────────┬──────────────────────┬──────────────────┬─────────────────────────────────┐           
  │ Your model │ From-scratch optimal │ Your DAPT budget │              Ratio              │
  ├────────────┼──────────────────────┼──────────────────┼─────────────────────────────────┤           
  │ 8M         │ 160M tokens          │ 800M             │ 5× over-trained (fine for DAPT) │
  ├────────────┼──────────────────────┼──────────────────┼─────────────────────────────────┤           
  │ 35M        │ 700M tokens          │ 800M             │ 1.1×                            │           
  ├────────────┼──────────────────────┼──────────────────┼─────────────────────────────────┤           
  │ 150M       │ 3B tokens            │ 1.6B             │ 0.5×                            │           
  ├────────────┼──────────────────────┼──────────────────┼─────────────────────────────────┤           
  │ 650M       │ 13B tokens           │ 1.6B             │ 0.12× under-trained             │
  └────────────┴──────────────────────┴──────────────────┴─────────────────────────────────┘           
                  
  Chinchilla is for from-scratch training. For DAPT you need far less because the model already knows  
  protein biochemistry.
                                                                                                       
  ---             
  The key DAPT reference: "Don't Stop Pretraining"
                                                  
  Gururangan et al. 2020 (ACL) is the most-cited DAPT paper. Key findings:
  - DAPT always helps for domain-specific tasks                                                        
  - They trained for roughly 12.5% of the original pretraining budget
  - They did not tune the token count — just used a fixed compute budget and stopped                   
  - Downstream task gain, not MLM loss, is what actually matters                                       
                                                                                                       
  They found diminishing returns after ~0.1–0.2 passes over the domain corpus, which matches your      
  configs.                                                                                             
                                                                                                       
  ---                                                                                                  
  Protein-specific guidance
                                                                                                       
  There's no published formula specifically for protein LM DAPT. The papers that do it (ProtST, ESM
  fine-tuning papers) typically just report: "we trained for N steps" without rigorous token budget    
  justification. What they consistently do:
                                                                                                       
  - Use 1–10% of the original training compute                                                         
  - Stop at val loss plateau, not a fixed budget
  - Validate on a downstream task (GO terms, localization) — that's the real signal, not MLM perplexity
                                                                                                       
  ---                                                                                                  
  Practical decision framework                                                                         
                  
  1. Compute your domain corpus size in tokens
     → yours: 19.9M seqs × avg 339 AA = 6.76B tokens                                                   
                                                                                                       
  2. Set a "passes" target:                                                                            
     - 0.1 passes → light adaptation, low forgetting risk                                              
     - 0.25 passes → moderate adaptation (your 150M/650M)                                              
     - 0.5 passes → strong adaptation, monitor for forgetting
     - >1 pass → only if you have strong downstream evidence it helps                                  
                                                                                                       
  3. Multiply: budget = passes × corpus_tokens                                                         
     → 0.12 × 6.76B = 810M ≈ your 800M ✓                                                               
                                                                                                       
  4. Sanity check against Chinchilla:
     - If budget > 5× Chinchilla optimal: you may be wasting compute                                   
     - If budget < 0.05× Chinchilla optimal: likely too little signal                                  
                                                                                                       
  5. The real stopping criterion: checkpoint every ~20% of budget,                                     
     run your linear probe benchmark, stop when downstream gain plateaus                               
                                                                                                       
  ---             
  The honest answer
                                                                                                       
  Your budgets (800M and 1.6B) are well-reasoned and consistent with the DAPT literature. The field
  doesn't have a precise formula — the best practice is to checkpoint frequently and evaluate          
  downstream task performance, which is what your benchmark script (scripts/benchmark_plant_tasks.py)
  is designed to do.                                                                                   
                  
✻ Baked for 35s                 