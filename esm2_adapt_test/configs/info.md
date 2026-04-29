  esm2_adapt_test/
  ├── configs/                                                                                                               
  │   ├── 8m.yaml       ← facebook/esm2_t6_8M_UR50D                                                                          
  │   ├── 35m.yaml      ← facebook/esm2_t12_35M_UR50D                                                                        
  │   └── 150m.yaml     ← facebook/esm2_t30_150M_UR50D                                                                       
  ├── results/          ← JSON files written here after each run                                                             
  ├── test_adapted_esm2.py                                                                                                   
  └── benchmark_plant_tasks.py                                                                                               
                                                                                                                             
  Usage for any model — run from project root:                                                                               
                                                                                                                           
  # Sanity test (perplexity, fill-mask, embeddings)                                                                          
  python esm2_adapt_test/test_adapted_esm2.py --config esm2_adapt_test/configs/35m.yaml                                      
                                                                                                                             
  # Full benchmark (localization + GO-term linear probe)                                                                     
  python esm2_adapt_test/benchmark_plant_tasks.py --config esm2_adapt_test/configs/35m.yaml                                  
                                                                                                                             
  # Quick smoke-test with fewer proteins                                                                                   
  python esm2_adapt_test/benchmark_plant_tasks.py --config esm2_adapt_test/configs/35m.yaml --max-proteins 500               
                                                                                                                             
  Results are saved as results/esm2_35m_sanity_2026-04-29.json and results/esm2_35m_benchmark_2026-04-29.json — one file per 
  model per day, so runs don't overwrite each other.                                                                         
                                                                                                                             
✻ Worked for 2m 52s                                                                                                        
