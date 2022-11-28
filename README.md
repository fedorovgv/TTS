
## FastSpeech2

--- 

### Instalation

This includes downloading LJSpeech, Mel Specs, Aligments, downloading pip libraries, creating normalized pitches and energies, WaveGlow checkpoint, pre trained model checkpoint.

```bash
python setup.py
```

All data for train available in `/dataset` directory:

1. alignments - ljspeech alignments
2. energies - calculated energies
3. energies_norm - normalized energies (by standart scaler)
4. LJSpeech - LJSpeech dataset
5. mels - mel specs
6. pitches - calculated pithes 
7. pitches_norm - normalized pitches (by standart scaler)
8. train.txt - LJSpeech transcript

Model checkpoint will be available in `./model/test/` folder. 

--- 

### Train

```bash
python train.py
```

Additional command line arguments:

```bash
python train.py \
   --version wandb_version \
   --project wandb_project \
   --batch-size 16
```

--- 

### Test

```bash
python test.py \
   --model-path ./model_ckpt/model.ckpt 
```

Additional command line arguments:

```bash
python test.py \
   --a 1.0 \
   --b 1.0 \
   --g 1.0 \
   --model-path ./model_ckpt/model.ckpt 
```
