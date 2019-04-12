testing_data=$1
output_file=$2

wget 'https://www.dropbox.com/s/hmu45vyzi5hpiah/dri_model10.pkl?dl=1'
wget 'https://www.dropbox.com/s/sr9pwti5iqnu16j/dri_model9.pkl?dl=1'
wget 'https://www.dropbox.com/s/xprgn5vbxoe255m/dri_model8.pkl?dl=1'
wget 'https://www.dropbox.com/s/f27rrrbq2zy47xq/dri_model7.pkl?dl=1'
wget 'https://www.dropbox.com/s/g40y69h68uc2v15/dri_model6.pkl?dl=1'
wget 'https://www.dropbox.com/s/4mgfmt3mgueummx/dri_model5.pkl?dl=1'
wget 'https://www.dropbox.com/s/imz8l6ywnqoegi2/dri_model4.pkl?dl=1'
wget 'https://www.dropbox.com/s/0blsz6n7n5c0uio/dri_model3.pkl?dl=1'
wget 'https://www.dropbox.com/s/55swsjjwchn8n9f/dri_model2.pkl?dl=1'
wget 'https://www.dropbox.com/s/5xz0jqdd7qjbt7v/dri_model1.pkl?dl=1'
python3 vote_test.py $1 $2 