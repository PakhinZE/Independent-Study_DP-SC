# Independent Study
model training for comparison between differential privacy model and normal model (DP vs non-DP)
## Step
use uv (python package and project manager)
download dataset first (read in /dataset) \
subsample size = 100000 (set DATA_SIZE in base_score.py, *SGD.py)
1. clean.py
2. vocab.ipynb
3. count.py (count token and sentence)
4. base_score.py (get base score for CharErrorRate)
5. Model
    * SGD.py
    * DP-SGD.py
    * Mask_SGD.py
    * Mask_DP-SGD.py
6. Get-result.ipynb (get result text file in /result)
7. Character Error Rate.ipynb (get score for CharErrorRate)
8. /GLEU/compute_gleu (read in /Note/GLEU)