# degenerate-detector

Uses a TensorFlow model that tries to classify an image as furry art, anime art, or neither.

## Installation and use
- Install Python 3 and pip
- Run `pip install -r requirements.txt` to install the requirements
- Place an image [PNG, JPG, or GIF] in `test` and run `python classify.py`

### Pretrained model info
- Furry model was generated from ~6000 most recent posts on e621 [from 7/12/2019] with ratings `safe/questionable` and tags `anthro, fur`
- Anime model was generaged from ~2000 most recent posts from r/animeart and r/awwnime [~1000 each, from 8/12/2019]
- "something non degenerate" model was generated from ~1000 most recent posts from  r/pics [8/12/2019]

### Retraining
Retraining is simple.
- Create a directory to store folders of images used for training, for example `dataset`.
```
├───dataset
   ├───anime_art
   ├───furry_art
   └───something_not_degenerate
```
Your tree should look like this.
- Acquire images for each category and place them in their respective folders
- Run `retrain.py --image_dir=dataset`
- Copy `/tmp/output_graph.pb` and `/tmp/output_graph.txt` to the `trained_model` folder and you're good to go! Simply place an image in the `test` directory and run `python classify.py` to run your image through the system.