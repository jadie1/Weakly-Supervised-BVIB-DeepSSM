# Weakly-Supervised-BVIB-DeepSSM
Weakly Supervised Bayesian Shape Modeling from Unsegmented Medical Images

To run the code first download the liver data from here: [Liver Data](https://drive.google.com/file/d/1RmhGj0ysbovOtc25wPMD-9AvX3L8rHeR/view?usp=sharing)and place it in the `data/` folder.

The run training by calling 'trainer.py' with a specificed config file, for example:
```
python trainer.py -c cfgs/full_supervision_vib.json
```
This will write the model, logged info, and a copy of the config file to a folder in `experiments/`, such as `experiments/liver/vib__vib_burnin__noise/`.

To run inference, call `eval.py` with the config file, for example:
```
python eval.py -c experiments/liver/vib__vib_burnin__noise/full_supervision_vib.json
```
This will write the predicted correspondence points and uncertainty values to the experiment directory. 