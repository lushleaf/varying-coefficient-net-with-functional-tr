# Simulated dataset
# ------------------------------------------------------------------------------------

# generate simulated dataset
python simu1_generate_data.py --save_dir dataset/simu1 --num_eval 100 --num_tune 20

# train and eval
# -- to run all models/methods in numbers of datasets
python main_batch.py --data_dir dataset/simu1 --save_dir save_dir --num_dataset num_dataset

# -- to run a singe run of models/methods with one dataset
# -- set --plt_adrf True for plot adrf curve
python main.py --data_dir dataset/simu1/eval/0 --save_dir save_dir --plt_adrf True

# ihdp dataset
# ------------------------------------------------------------------------------------

# generate ihdp dataset
python ihdp_generate_data.py

# train and eval
# -- to run all models/methods in numbers of datasets
python main_batch_ihdp.py

# -- to run a singe run of models/methods with one dataset
# -- set --plt_adrf True for plot adrf curve
python main_ihdp.py --plt_adrf True

# news dataset
# ------------------------------------------------------------------------------------
# generate ihdp dataset
python news_generate_data.py

# train and eval
# -- to run all models/methods in numbers of datasets
python main_batch_news.py

# -- to run a singe run of models/methods with one dataset
# -- set --plt_adrf True for plot adrf curve
python main_news.py --plt_adrf True