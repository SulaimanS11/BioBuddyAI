## BioBuddyAI

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