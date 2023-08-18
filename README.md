# Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution


[Our Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lu_Learning_Spatial-Temporal_Implicit_Neural_Representations_for_Event-Guided_Video_Super-Resolution_CVPR_2023_paper.pdf)

Please feel free to contact me if you have any issues. (ylu066"at"connect.hkust-gz.edu.cn)


<!-- [video](https://www.youtube.com/watch?v=ty531p2Me7Qng) -->
[YouTuBe Video](https://www.youtube.com/watch?v=ty531p2Me7Qng)

## Training or Testing

Here is an example of training. The entry point for all training and testing is ‘main.py’.

```sh
sh options/CED/egisr-alpx-continue-event-skip-connection-a5.sh
```

## Dataset

Our dataset is at this link. If you are interested, please download it.

```
LINK: https://pan.baidu.com/s/1_J9rNdhEjEsMSLrv5biflg
Code: cvpr
```

Because the data set takes up too much storage, I have compressed the dataset to reduce storage bandwidth and facilitate your download.
Specifically, I compressed the events into a ProtoBuf.
The description file of pb is `tools/1-CompressDataset/event_frame.proto`.
For more information about the use of pb, please refer to the documentation (https://protobuf.dev).
Read the pb file, please refer to the code `tools/1-CompressDataset/read_pb.py`
During training, converting pb to numpy files will speed up training.

## Citations

If this work is helpful for your research, please consider citing the following BibTeX entry.

```sh
@inproceedings{lu2023learning,
  title={Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution},
  author={Lu, Yunfan and Wang, Zipeng and Liu, Minjie and Wang, Hongjian and Wang, Lin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1557--1567},
  year={2023}
}
```

## Acknowledged items

Thanks to these two open source projects. I also hope my project can help you.

[ESRT](https://github.com/luissen/ESRT)
[MMP-RNN](https://github.com/sollynoay/MMP-RNN)