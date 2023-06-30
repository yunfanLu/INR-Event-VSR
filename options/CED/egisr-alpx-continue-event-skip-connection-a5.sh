export CUDA_VISIBLE_DEVICES="7"
export PYTHONPATH="./":PYTHONPATH

python egvsr/main.py \
  --yaml_file="./options/CED/Compare/C2-Alpx-Compare/egisr-alpx-continue-event-skip-connection-a5.yaml" \
  --log_dir="./log/CED/Compare/C2-Alpx-Compare/egisr-alpx-continue-event-skip-connection-a5/" \
  --alsologtostderr=True