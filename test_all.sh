export CUDA_VISIBLE_DEVICES="7"
export PYTHONPATH="./":$PYTHONPATH


echo "Dataset Testing"

echo "Tools"

echo "Analysis"

#python egvsr/datasets/alpx_vsr_dataset_test.py
#python egvsr/datasets/alpx_vsr_dataset_foreach_test.py
#python egvsr/datasets/alpx_vsr_dataset_visuallize_test.py
python egvsr/models/rssrt/random_scale_super_resolution_with_event_test.py