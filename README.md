# FragmentVC

### Preprocessing

You can preprocess multiple corpora at the same time.
But the corpus paths should be the directory that contains the speaker directories.
```bash
./preprocess.py \
    <CORPORA_DIR>/VCTK-Corpus/wav48 \
    <CORPORA_DIR>/LibriSpeech/train-clean-360 \
    ...
    features  # processed data output directory
```

After preprocessing, you will see these in the output directory, e.g. `features`:
```text
metadata.json
utterance-000x7gsj.npz
utterance-00wq7b0f.npz
utterance-01lpqlnr.npz
...
```

### Training 

```bash
python3 train.py \
    --save_dir ./ckpts \
    --preload \ # load all data into memory 
    --comment <COMMENT> \ # name of log directory
    ...
    features # dir of processed data, containing metadata.json
```

It is recommended to specify `--preload` for boosting training.
If `--comment` is specified, the logging will be stored in ./logs/<COMMENT>.

### Inference

You can convert audio one by one.

```bash
python3 convert.py \
    --wav2vec_path checkpoints/wav2vec_small.pt \
    --vocoder_path checkpoints/vocoder.pt \
    --ckpt_path <CKPT_PATH> \
    ...
    <SOURCE_PATH> \ # ex: p225/p225_001.wav
    <REFERENCE_PATHS> \ # ex: p226/p226_001.wav p226/p226_002.wav p226/p226_003.wav ...
    <OUTPUT_DIR>
```

```bash
python3 convert_batch.py \
    --wav2vec_path checkpoints/wav2vec_small.pt \
    --vocoder_path checkpoints/vocoder.pt \
    --ckpt_path <CKPT_PATH> \
    ...
    <INFO_PATH> \ # path to information file for inference
    <OUTPUT_DIR>
```

```text
The format of information file would be like
{
    <PATH_NAME>: {"source": <SOURCE_PATH>, "target": <LIST_OF_TARGET_PATHS>}
    ...
}
```
