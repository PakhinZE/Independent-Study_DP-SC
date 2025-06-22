# Independent Study
model training for comparison between differential privacy model and normal model (DP vs non-DP)
## Step
use uv (python package and project manager)
download dataset first (read in /dataset) \
subsample size = 100000 (set DATA_SIZE in base_score.py, *SGD.py)
1. clean.py
2. create_vocab.py
3. count.py (count token and sentence)
4. base_score.py (get base score for CharErrorRate)
5. fix opacus code (read in /Note/Fix Opacus.txt)
6. Model
    * SGD.py
    * DP-SGD.py
    * Mask_SGD.py
    * Mask_DP-SGD.py
7. get_result_text.py (get result text file in /result)
8. get_character_error_rate.py (get score for Character Error Rate)
9. /GLEU/compute_gleu (read in /Note/GLEU.txt)