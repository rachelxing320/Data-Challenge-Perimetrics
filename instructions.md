This exercise is designed to evaluate your ability to take multiple pieces of working code and "glue" them together.

1. Download the following notebooks:
	* https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/images/transfer_learning_with_hub.ipynb
	* https://storage.googleapis.com/tensorflow_docs/docs/site/en/tutorials/keras/classification.ipynb

2. Set up a Python 3 environment with the following packages:
	```bash
	$ pip install matplotlib pillow jupyterlab
	```

3. **MINIMALLY** modify `transfer_learning_with_hub.ipynb` so that it uses the MNIST fashion data set. Insert as much code as is appropriate from `classification.ipynb` into `transfer_learning_with_hub.ipynb`. In addition, fewer than a dozen lines of code in `classification.ipynb` will need to be changed to make this work. Please note the following:

	* Mark all added/changed/deleted lines of code in `classification.ipynb` (including code coming from other files or these instructions) with a comment including the word `MODIFIED` in all caps. These are the lines of code we will evaluate.

	* You may find the following helper functions useful to incorporate:

		```python
		def convert_to_color(images, size=(224,224)):
			images_2 = np.stack((images,)*3, axis=-1)
			images_3 = tf.image.resize(
				images_2, 
				size, 
				method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
			)
			return images_3

		def convert_to_one_hot(labels):
			rows = np.arange(labels.size)
			one_hot = np.zeros((labels.size, labels.max()+1))
			one_hot[rows, labels] = 1
			return one_hot
		```

	* When training, use the following hyperparameters: `batch_size=64` and `steps_per_epoch=50`

	* Towards the end of the notebook, predictions are checked and visualized as a table of images. There is no need to check more than the first 30 test images. For reference, please see the `example_prediction_grid.png` output image. Your result may be better or worse than this, but your model should be trained to achieve at least 70% accuracy.

4. Submit your final `transfer_learning_with_hub.ipynb` file. No other files should be submitted.