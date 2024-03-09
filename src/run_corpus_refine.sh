nohup python -u corpus_selection.py 42 0 0-91217223-20 ml-100k > logs/corpus_selection_ml100k_gpu0.log &
nohup python -u corpus_selection.py 42 1 91217223-182434446-20 ml-100k > logs/corpus_selection_ml100k_gpu1.log &
nohup python -u corpus_selection.py 42 2 182434446-273651669-20 ml-100k > logs/corpus_selection_ml100k_gpu2.log &
nohup python -u corpus_selection.py 42 3 273651669-364868892-20 ml-100k > logs/corpus_selection_ml100k_gpu3.log &

nohup python -u corpus_selection.py 42 0 0-91217223-20 coupon >> logs/corpus_selection_coupon_gpu0.log &
nohup python -u corpus_selection.py 42 1 91217223-182434446-20 coupon >> logs/corpus_selection_coupon_gpu1.log &
nohup python -u corpus_selection.py 42 2 182434446-273651669-20 coupon >> logs/corpus_selection_coupon_gpu2.log &
nohup python -u corpus_selection.py 42 2 273651669-364868892-20 coupon >> logs/corpus_selection_coupon_gpu3.log &

nohup python -u corpus_selection.py 42 0 0-1614668-20 mexico_restaurant >> logs/corpus_selection_restaurant_gpu0.log &
nohup python -u corpus_selection.py 42 1 1614668-3229336-20 mexico_restaurant >> logs/corpus_selection_restaurant_gpu1.log &
nohup python -u corpus_selection.py 42 2 3229336-4844004-20 mexico_restaurant >> logs/corpus_selection_restaurant_gpu2.log &
nohup python -u corpus_selection.py 42 3 4844004-6458670-20 mexico_restaurant >> logs/corpus_selection_restaurant_gpu3.log &
