testing_data=$1
output_file=$2

wget 'https://www.dropbox.com/s/hmu45vyzi5hpiah/dri_model10.pkl?dl=1'
python3 Saliency.py $1 $2
python3 pre_lime.py $1 $2
python3 Lime.py $1 $2
python3 visualization.py $1 $2 