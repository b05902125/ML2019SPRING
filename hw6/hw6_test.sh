testing_data=$1
output_file=$2

wget 'https://www.dropbox.com/s/ad1yzmvzoqtm8zs/drp_ckpt_74.842?dl=1'
wget 'https://www.dropbox.com/s/stelvneiiowa7ig/drp_ckpt_75.116?dl=1'
wget 'https://www.dropbox.com/s/gvmuzoq6az6cb5s/new_ckpt_75.072?dl=1'
wget 'https://www.dropbox.com/s/tqcdgclmuma0jqe/new_ckpt_75.266?dl=1'
wget 'https://www.dropbox.com/s/mfjh95zal4klfdn/new_word2vec?dl=1'

python3 hw6_test.py $1 $2 $3 