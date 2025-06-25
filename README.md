## BioBuddyAI

### Notes for now: 
**1. Train the Model**

Run this command from the root of your project:

```bash
python train.py
```

This will Train the model from scratch, save the weights to `poison_model.pth`
You’ll see output like:

```python-repl
Epoch 1 Loss: ...
Epoch 2 Loss: ...
```

Once it's done, you should see `poison_model.pth` appear in the folder.

**2. Now Run the Classifier Again**
Now your main.py will work correctly:

```bash
python main.py
```

### Steps to use for judging:

1. Say: “Eastern Massasauga”
2. Pi scans a plant/snake using open cv
3. Pi says: “This is Poisonous. Critical Threat. Stay away.” using quantum networks!
4. Print/Show histogram
5. Show logging dashboard or heatmap
6. Ask it: “Tell me more” → it speaks fun facts from `plant-snake-facts.json`
7. Show code using Qiskit + live quantum backend


### To train the model
Simply run the following command in the terminal/console:
```bash
python train_model.py
```