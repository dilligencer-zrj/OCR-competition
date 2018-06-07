运行环境：python 3.5；tensorflow环境到tensorflow的官方github地址下载  tf_nightly_gpu-1.head-cp35-cp35m-linux_x86_64 版本



1.将训练集图片放在train目录下，同时将label文件以txt文本形式保存在train目录下；test 同样如此。


2.train
    cd Attention-OCR
    python src/launcher.py --phase=train --data-path=train/train.txt --data-base-dir=train --log-path=log.txt --no-load-model

3.test
    python src/launcher.py --phase=test --visualize --data-path=evaluation_data/svt/test.txt --data-base-dir=evaluation_data/svt --log-path=log.txt --load-model --model-dir=model --output-dir=results

4.产生csv文件

    cd  Attention-OCR/data_preprocess
    python csv_to_txt.py

注意更改代码中文件的目录