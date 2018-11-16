# Deep Flare Net (DeFN) Astrophysical Journal 2018 version

Komei Sugiura,
National Institute of Information and Communications Technology, Japan

## 0. License

* BSD 3-Clause Clear License

## 1. Prerequisite

* Ubuntu 16.04 or 14.04
* Python 3.4.3

## 2. Install

* In the following procedure, `~/work` is assumed to be used as a working directory.

```
$ cd ~/work/
$ git clone https://github.com/komeisugiura/defn18.git
$ cd defn18
$ pip install -r requirements.txt
$ pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp34-cp34m-linux_x86_64.whl
```

## 3. Download data

* Visit `http://wdc.nict.go.jp/IONO/wdc/solarflare/index.html` and download `defn_feature_database_v1_pl.zip`.

```
$ cd ~/work/
$ mv defn_feature_database_v1_pl.zip ./
$ unzip defn_feature_database_v1_pl.zip
(password is required)
$ ln -s defn_feature_database_v1/defn_input_database/charval2017X_*.csv.gz ./defn18/data/
```

## 4. Run

```
$ cd ~/work/src
$ ./01RUN_deepflarenet.sh
```

* The following result will be shown. This means that TSS=0.8024 is obtained by using a pretrained model.

`[008000]Acc: Tra=0.8345, Val=0.8584, Tes=0.8584, MaxVal=0.8584(0.8584), TSS=0.8024`

## 5. Training DeFN from scratch

Modify `src/deepflarenet.py`.

* Uncomment the following line to train the model
`# net1.train_model(update_interval=100)`

* Uncomment the following line to save the trained model. Current model is overwritten.
`# net1.save_model(myflag.outfile_model)`

* Comment the following two lines out, if you don't like to load the model
```
net1.load_model(myflag.infile_model)
net1.show_training_status(epoch=8000)
```

## A. References

1. N. Nishizuka, K. Sugiura, Y. Kubo, M. Den, and M. Ishii, "Deep Flare Net (DeFN) Model for Solar Flare Prediction", The Astrophysical Journal, Vol. 858, Issue 2, 113 (8pp), 2018. DOI: 10.3847/1538-4357/aab9a7

# Local Variables:
# coding: utf-8
# End:
