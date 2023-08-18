export CUDA_VISIBLE_DEVICES="6"
export PYTHONPATH="./":PYTHONPATH

python egvsr/main.py \
  --yaml_file="./options/CED/Compare/T1-CED-Compare/egisr-ceds-continue-4x-addchannel-t4.yaml" \
  --log_dir="./log/CED/Compare/T1-CED-Compare/egisr-ceds-continue-4x-addchannel-t4/" \
  --alsologtostderr=True